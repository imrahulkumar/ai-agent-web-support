import axios from 'axios';
import * as cheerio from 'cheerio';
import OpenAI from "openai";
import dotenv from 'dotenv';
import { ChromaClient } from "chromadb";

dotenv.config();
const openai = new OpenAI({
       apiKey: process.env.OPENAI_API_KEY,
       baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/",
       timeout: 60 * 1000,
       maxRetries: 1,
});
const dhlUrl = 'https://www.gbu.ac.in';
const chromaClient = new ChromaClient({
    host: 'localhost',
    port: 8000,
});

const WEB_COLLECTION = 'WEB_SCAPED_DATA_COLLECTION-1';
const CHAT_MODEL = 'gemini-3.5-flash';
const MAX_PAGES = 10;

const visited = new Set();


async function scapeWebpage(url){
  try {
    console.log(`🌐 Scraping: ${url}`);
    const { data } = await axios.get(url, { timeout: 15 * 1000 });
    const $ = cheerio.load(data);

    const pageHead = `${$('title').text()} ${$('meta[name="description"]').attr('content') || ''}`.trim();
    $('script, style, noscript').remove();
    const pageBody = $('body').text().replace(/\s+/g, ' ').trim();
    const internalLink = [];
    const externalLink = [];

    $('a').each((index, element) => {
      const href = $(element).attr('href');
      if (!href) return;
      let link;
      try {
        link = new URL(href, url);
      } catch {
        return;
      }
      link.hash = '';
      if (link.href.startsWith(dhlUrl)) {
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

async function ingest(url = '') {
    if (visited.has(url) || visited.size >= MAX_PAGES) return;
    visited.add(url);

    const page = await scapeWebpage(url);
    if (!page) return;
    const { head, body, internalLink } = page;

    if (head) {
        const headEmbedding = await generateVectorEmbeddings({ text: head });
        await insertIntoDB({ id: `${url}#head`, embedding: headEmbedding, url, body: head, head });
    }

    const bodyChunks = chunkText(body, 2000);
    console.log(`   🧩 ${bodyChunks.length} chunks to embed`);

    for (const [index, chunk] of bodyChunks.entries()) {
        const chunkEmbedding = await generateVectorEmbeddings({ text: chunk });
        await insertIntoDB({ id: `${url}#${index}`, embedding: chunkEmbedding, url, body: chunk, head });
    }
    console.log(`   💾 Stored ${url}`);

    for (const link of internalLink) {
        await ingest(link);
    }
}

function chunkText(text, chunkSize = 500) {
    const chunks = [];
    for (let i = 0; i < text.length; i += chunkSize) {
        chunks.push(text.slice(i, i + chunkSize));
    }
    return chunks;
}

async function generateVectorEmbeddings({text}) {

    const embedding = await openai.embeddings.create({
        model: "gemini-embedding-001",
        input: text,
        encoding_format: "float",
    });

    return embedding.data[0].embedding;
}

async function insertIntoDB({ id, embedding, url, body = '', head = '' }) {
    const collection = await chromaClient.getOrCreateCollection({ name: WEB_COLLECTION});
    await collection.upsert({
        ids: [id],
        embeddings: [embedding],
        metadatas: [{ url, body, head }],
    });
}


async function chat(question = '') {
   console.log(`\n❓ Question: ${question}`);
   const questionEmbedding = await generateVectorEmbeddings({ text: question });
    const collection = await chromaClient.getOrCreateCollection({ name: WEB_COLLECTION });
   const collectionResult = await  collection.query({
       queryEmbeddings: [questionEmbedding],
       nResults: 5
   });

   const metadatas = collectionResult.metadatas[0].filter(Boolean);
   console.log(`🔎 Retrieved ${metadatas.length} chunks from Chroma`);

   const body  = metadatas.map(metadata => metadata.body).filter(e => e && e.trim());

   const urls = [...new Set(metadatas.map(metadata => metadata.url).filter(e => e && e.trim()))];
   console.log(`🔗 Source URLs: ${urls.join(', ')}`);

   console.log(`💬 Asking ${CHAT_MODEL}...`);
   const response = await openai.chat.completions.create({
       model: CHAT_MODEL,
       messages: [
           { role: "system", content: `You are an AI support agent expert in providing support to user on behalf a webpage. Given the context about page content, reply the user accordingly` },
           { role: "user",
             content: `
              Query: ${question}\n\n
              URLs: ${urls.join(", ")}\n\n
              Retrived Context: ${body.join(", ")}
             `
           }
       ]
   });

   console.log('🤖 ->', response.choices[0].message.content);
}


async function main() {
    console.log('🫀 Chroma heartbeat:', await chromaClient.heartbeat());

    const collection = await chromaClient.getOrCreateCollection({ name: WEB_COLLECTION });
    const count = await collection.count();
    console.log(`📦 Collection "${WEB_COLLECTION}" has ${count} records`);

    if (count === 0) {
        console.log(`🚀 Ingesting ${dhlUrl} (max ${MAX_PAGES} pages)...`);
        await ingest(dhlUrl);
        console.log(`✅ Ingestion done, ${visited.size} pages visited`);
    }

    await chat('what course are avillable in university');
}

main().catch((error) => {
    console.error('❌ Failed:', error.status ?? '', error.message);
    process.exit(1);
});
