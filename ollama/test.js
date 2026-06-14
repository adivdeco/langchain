import { ChatOllama } from "@langchain/ollama";
import "dotenv/config";

async function runOllamaCloud() {
  try {
    const llm = new ChatOllama({
      model: "nemotron-3-ultra:cloud",

    });

    console.log("Sending prompt to Ollama Cloud...");

    const response = await llm.invoke("what is the capital of France?");

    console.log("\nResponse:");
    console.log(response.content);

  } catch (error) {
    console.error("Error communicating with Ollama Cloud:", error.message);
  }
}

runOllamaCloud();