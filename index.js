/**
 * AI Agent – Web Support
 * ----------------------
 * A small RAG (Retrieval-Augmented Generation) support bot for a website.
 *
 * How it works (in order):
 *   1. SCRAPE  – Crawl the target website (same-domain links only, up to MAX_PAGES).
 *   2. CHUNK   – Split each page's text into fixed-size chunks.
 *   3. EMBED   – Turn each chunk into a vector using Gemini's embedding model.
 *   4. STORE   – Save vectors + text into a ChromaDB collection.
 *   5. CHAT    – For a user question: embed it, find the closest chunks in Chroma,
 *                and ask the Gemini chat model to answer using those chunks as context.
 *
 * Steps 1–4 (ingestion) only run when the collection is empty, so restarts are cheap.
 */

// ============================================================================
// 1. Imports
// ============================================================================
import axios from 'axios';               // HTTP client to download web pages
import * as cheerio from 'cheerio';      // jQuery-like HTML parser
import OpenAI from 'openai';             // OpenAI SDK (pointed at Gemini's OpenAI-compatible API)
import dotenv from 'dotenv';             // Loads variables from .env into process.env
import { ChromaClient } from 'chromadb'; // Vector database client

// Load .env before anything reads process.env
dotenv.config();

// ============================================================================
// 2. Configuration
// ============================================================================

// Website to crawl. Only links that start with this URL are followed.
const TARGET_URL = 'https://www.gbu.ac.in';

// Chroma collection that holds the scraped + embedded content.
// Changing this name starts a fresh collection (and triggers a new crawl).
const WEB_COLLECTION = 'WEB_SCAPED_DATA_COLLECTION-1';

// Gemini models (accessed through the OpenAI-compatible endpoint)
const CHAT_MODEL = 'gemini-3.5-flash';
const EMBEDDING_MODEL = 'gemini-embedding-001';

const MAX_PAGES = 10;         // Max number of pages to crawl
const CHUNK_SIZE = 2000;      // Characters per text chunk stored in Chroma
const TOP_K_RESULTS = 5;      // Number of chunks retrieved per question

// ============================================================================
// 3. Clients
// ============================================================================

// The OpenAI SDK is used as-is, but requests go to Google's Gemini API.
// OPENAI_API_KEY in .env must therefore contain a Gemini API key.
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: 'https://generativelanguage.googleapis.com/v1beta/openai/',
    timeout: 60 * 1000,
    maxRetries: 1,
});

// ChromaDB running locally (see docker-compose.yml)
const chromaClient = new ChromaClient({
    host: 'localhost',
    port: 8000,
});

// URLs already crawled in this run – prevents loops and duplicate work
const visited = new Set();

// ============================================================================
// 4. Scraping
// ============================================================================

/**
 * Download a single page and extract its useful parts.
 *
 * @param {string} url - Page to scrape.
 * @returns {Promise<{head: string, body: string, internalLink: string[], externalLink: string[]} | null>}
 *          `head`  – page <title> + meta description
 *          `body`  – visible text of the page, whitespace collapsed
 *          `internalLink` / `externalLink` – absolute links found on the page
 *          Returns null if the page could not be fetched.
 */
async function scrapeWebpage(url) {
    try {
        console.log(`🌐 Scraping: ${url}`);
        const { data } = await axios.get(url, { timeout: 15 * 1000 });
        const $ = cheerio.load(data);

        // Title + meta description give a short summary of the page
        const pageHead = `${$('title').text()} ${$('meta[name="description"]').attr('content') || ''}`.trim();

        // Drop non-visible content, then take the readable text of <body>
        $('script, style, noscript').remove();
        const pageBody = $('body').text().replace(/\s+/g, ' ').trim();

        // Collect every link and split into internal (same site) vs external
        const internalLink = [];
        const externalLink = [];

        $('a').each((index, element) => {
            const href = $(element).attr('href');
            if (!href) return;

            // Resolve relative links (e.g. "/about") against the current page URL
            let link;
            try {
                link = new URL(href, url);
            } catch {
                return; // Skip malformed hrefs
            }
            link.hash = ''; // "#section" anchors point to the same page

            if (link.href.startsWith(TARGET_URL)) {
                internalLink.push(link.href);
            } else {
                externalLink.push(link.href);
            }
        });

        console.log(`   ✔ head: ${pageHead.length} chars, body: ${pageBody.length} chars, internal links: ${internalLink.length}, external links: ${externalLink.length}`);
        return { head: pageHead, body: pageBody, internalLink, externalLink };
    } catch (error) {
        console.error(`   ✘ Error scraping ${url}:`, error.message);
        return null;
    }
}

// ============================================================================
// 5. Text processing & embeddings
// ============================================================================

/**
 * Split text into fixed-size pieces so each one fits the embedding model.
 *
 * @param {string} text
 * @param {number} chunkSize - Max characters per chunk.
 * @returns {string[]}
 */
function chunkText(text, chunkSize = 500) {
    const chunks = [];
    for (let i = 0; i < text.length; i += chunkSize) {
        chunks.push(text.slice(i, i + chunkSize));
    }
    return chunks;
}

/**
 * Convert text into a vector (array of numbers) that captures its meaning.
 * Similar texts produce vectors that are close together, which is what
 * lets Chroma find relevant chunks for a question.
 *
 * @param {{text: string}} params
 * @returns {Promise<number[]>}
 */
async function generateVectorEmbeddings({ text }) {
    const embedding = await openai.embeddings.create({
        model: EMBEDDING_MODEL,
        input: text,
        encoding_format: 'float',
    });

    return embedding.data[0].embedding;
}

// ============================================================================
// 6. Vector database (ChromaDB)
// ============================================================================

/**
 * Insert or update one record in the Chroma collection.
 * The original text is stored in metadata so it can be shown to the LLM later.
 *
 * @param {object} params
 * @param {string} params.id        - Unique id, e.g. "<url>#head" or "<url>#3"
 * @param {number[]} params.embedding
 * @param {string} params.url       - Page the text came from
 * @param {string} [params.body]    - The text chunk itself
 * @param {string} [params.head]    - Page title/description, kept for context
 */
async function insertIntoDB({ id, embedding, url, body = '', head = '' }) {
    const collection = await chromaClient.getOrCreateCollection({ name: WEB_COLLECTION });
    await collection.upsert({
        ids: [id],
        embeddings: [embedding],
        metadatas: [{ url, body, head }],
    });
}

// ============================================================================
// 7. Ingestion (scrape → chunk → embed → store, recursively)
// ============================================================================

/**
 * Crawl a page, store its content in Chroma, then follow its internal links.
 * Stops once MAX_PAGES pages have been visited.
 *
 * @param {string} url
 */
async function ingest(url = '') {
    // Skip pages already seen, and stop when the page limit is reached
    if (visited.has(url) || visited.size >= MAX_PAGES) return;
    visited.add(url);

    // Step 1: scrape
    const page = await scrapeWebpage(url);
    if (!page) return;
    const { head, body, internalLink } = page;

    // Step 2: store the page summary (title + description) as its own record
    if (head) {
        const headEmbedding = await generateVectorEmbeddings({ text: head });
        await insertIntoDB({ id: `${url}#head`, embedding: headEmbedding, url, body: head, head });
    }

    // Step 3: chunk the body text, embed each chunk and store it
    const bodyChunks = chunkText(body, CHUNK_SIZE);
    console.log(`   🧩 ${bodyChunks.length} chunks to embed`);

    for (const [index, chunk] of bodyChunks.entries()) {
        const chunkEmbedding = await generateVectorEmbeddings({ text: chunk });
        await insertIntoDB({ id: `${url}#${index}`, embedding: chunkEmbedding, url, body: chunk, head });
    }
    console.log(`   💾 Stored ${url}`);

    // Step 4: crawl linked pages on the same site (depth-first)
    for (const link of internalLink) {
        await ingest(link);
    }
}

// ============================================================================
// 8. Chat (retrieve relevant chunks → ask the LLM)
// ============================================================================

/**
 * Answer a question using the website content stored in Chroma.
 *
 * @param {string} question
 */
async function chat(question = '') {
    console.log(`\n❓ Question: ${question}`);

    // Step 1: embed the question so it can be compared with stored chunks
    const questionEmbedding = await generateVectorEmbeddings({ text: question });

    // Step 2: find the most similar chunks in Chroma
    const collection = await chromaClient.getOrCreateCollection({ name: WEB_COLLECTION });
    const collectionResult = await collection.query({
        queryEmbeddings: [questionEmbedding],
        nResults: TOP_K_RESULTS,
    });

    const metadatas = collectionResult.metadatas[0].filter(Boolean);
    console.log(`🔎 Retrieved ${metadatas.length} chunks from Chroma`);

    // Step 3: build the context – chunk texts plus the unique source URLs
    const body = metadatas.map(metadata => metadata.body).filter(e => e && e.trim());
    const urls = [...new Set(metadatas.map(metadata => metadata.url).filter(e => e && e.trim()))];
    console.log(`🔗 Source URLs: ${urls.join(', ')}`);

    // Step 4: ask the chat model to answer using only that context
    console.log(`💬 Asking ${CHAT_MODEL}...`);
    const response = await openai.chat.completions.create({
        model: CHAT_MODEL,
        messages: [
            {
                role: 'system',
                content: `You are an AI support agent expert in providing support to user on behalf a webpage. Given the context about page content, reply the user accordingly`,
            },
            {
                role: 'user',
                content: `
              Query: ${question}\n\n
              URLs: ${urls.join(', ')}\n\n
              Retrived Context: ${body.join(', ')}
             `,
            },
        ],
    });

    console.log('🤖 ->', response.choices[0].message.content);
}

// ============================================================================
// 9. Entry point
// ============================================================================

async function main() {
    // Make sure ChromaDB is reachable before doing anything else
    console.log('🫀 Chroma heartbeat:', await chromaClient.heartbeat());

    const collection = await chromaClient.getOrCreateCollection({ name: WEB_COLLECTION });
    const count = await collection.count();
    console.log(`📦 Collection "${WEB_COLLECTION}" has ${count} records`);

    // Only crawl the site the first time (when the collection is empty)
    if (count === 0) {
        console.log(`🚀 Ingesting ${TARGET_URL} (max ${MAX_PAGES} pages)...`);
        await ingest(TARGET_URL);
        console.log(`✅ Ingestion done, ${visited.size} pages visited`);
    }

    // Ask a sample question – change this to try other queries
    await chat('what course are avillable in university');
}

main().catch((error) => {
    console.error('❌ Failed:', error.status ?? '', error.message);
    process.exit(1);
});
