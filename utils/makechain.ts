import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the earnings call details and the following questions, rephrase the questions to be standalone questions.

Earnings Call Details:
{earnings_call_details}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `Earnings Call Analyzer for [COMPANY_NAME]

{context}

Question: {question}
Analyst Note in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-4', //change this to gpt-4 if you have access
    max_tokens: 8192, // Set the desired max token limit
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
