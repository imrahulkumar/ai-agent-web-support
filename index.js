import axios from 'axios';
import * as cheerio from 'cheerio';
import OpenAI from "openai";
import dotenv from 'dotenv';
import { ChromaClient } from "chromadb";

dotenv.config();
const openai = new OpenAI({
       apiKey: process.env.OPENAI_API_KEY,
       baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/",
});
const dhlUrl = 'https://www.gbu.ac.in';
const chromaClient = new ChromaClient({
    path: 'http://localhost:8000'
});
chromaClient.heartbeat()

const WEB_COLLECTION = 'WEB_SCAPED_DATA_COLLECTION-1';


async function scapeWebpage(url){
  try {
    const { data } = await axios.get(url);
    const $ = cheerio.load(data);

    const pageHead = $('head').html();
    const pageBody = $('body').html();
    const internalLink = [];
    const externalLink = [];

    $('a').each((index, element) => {
      const link = $(element).attr('href');
      if (link && link.startsWith(dhlUrl)) {
        internalLink.push(link);
      } else if (link) {
        externalLink.push(link);
      }
    });
    return { head: pageHead, body: pageBody, internalLink, externalLink };

  } catch (error) {
    console.error('Error scraping webpage:', error);
  }
}

async function ingest(url = '') {
     const { head, body, internalLink, externalLink } =  await scapeWebpage(url);
     const headEmbedding = await generateVectorEmbeddings({ text: head });
     await insertIntoDB({embedding: headEmbedding, head});
    const bodyChunks = chunkText(body, 2000);

    for (const chunk of bodyChunks) {
        const chunkEmbedding = await generateVectorEmbeddings({ text: chunk });
        // You can store or process the chunkEmbedding as needed
        await insertIntoDB({embedding:chunkEmbedding, url, body:chunk, head});
    }

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
        model: "text-embedding-3-small",
        input: text,
        encoding_format: "float",
    });

    return embedding.data[0].embedding;
}

async function insertIntoDB(embedding, url, body, head) {
    const collection = await chromaClient.getOrCreateCollection({ name: WEB_COLLECTION});
    await collection.add({
        embeddings: [embedding],
        metadatas: [{ url, body, head }],
        ids: [url]
    });
}

scapeWebpage(dhlUrl).then((text) => {
//   console.log(text);
});


async function chat(question = '') {
   const questionEmbedding = await generateVectorEmbeddings({ text: question });
    const collection = await chromaClient.getOrCreateCollection({ name: WEB_COLLECTION });
   const collectionResult = await  collection.query({
       query_embeddings: [questionEmbedding],
       n_results: 5
   });

   const body  = collectionResult.metadatas[0].map(metadata => metadata.body).filter(e => e.trim() || !!e);

   const urls = collectionResult.metadatas[0].map(metadata => metadata.url).filter(e => e.trim() || !!e);

   const response = await openai.chat.completions.create({
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


chat('what course are avillable in university');