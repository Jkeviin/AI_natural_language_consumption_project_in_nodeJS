import axios from "axios";
const HF_ACCESS_TOKEN = "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";

async function query(data) {
  const response = await axios.post(
    "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
    data,
    {
      headers: { Authorization: `Bearer ${HF_ACCESS_TOKEN}` },
    }
  );
  const result = response.data;
  return result;
}

async function query2(data) {
  const response = await axios.post(
    "https://api-inference.huggingface.co/models/xlm-roberta-large",
    data,
    {
      headers: { Authorization: `Bearer ${HF_ACCESS_TOKEN}` },
    }
  );
  const result = response.data;
  return result;
}

query({
  inputs: {
    source_sentence: "That is a happy person",
    sentences: [
      "That is a happy dog",
      "That is a very happy person",
      "Today is a sunny day",
    ],
  },
}).then((response) => {
  console.log(JSON.stringify(response));
});

query2({
  inputs: '<mask> Mundo!',
  options: {
    use_cache: true,
    wait_for_model: false
  }
}).then((response) => {
  console.log((response));
});