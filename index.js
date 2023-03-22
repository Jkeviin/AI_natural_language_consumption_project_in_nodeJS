import { HfInference } from '@huggingface/inference'

const HF_ACCESS_TOKEN = "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";

const hf = new HfInference(HF_ACCESS_TOKEN);

const fillMask = await hf.fillMask({
  model: 'bert-base-uncased',
  inputs: '[MASK] Mundo!',
  options: {
    use_cache: true,
    wait_for_model: false
  }
});
//console.log(fillMask)
//----------------------//

const summarization = await hf.summarization({
  model: 'facebook/bart-large-cnn',
  inputs: 'La torre tiene 324 metros de altura, aproximadamente la misma altura que un edificio de 81 plantas, y es la estructura más alta de París. Su base es cuadrada y mide 125 metros de lado. Durante su construcción, la Torre Eiffel superó al Monumento a Washington y se convirtió en la estructura artificial más alta del mundo, título que mantuvo durante 41 años, hasta que se terminó el Edificio Chrysler de Nueva York en 1930.',
  parameters: {
    min_length: 30,
    max_length: 100,
    top_k: 50,
    top_p: 0.95,
    temperature: 0.7,
    repetition_penalty: 1.5,
    max_time: 30,
  },
  options: {
    use_cache: true,
    wait_for_model: false,
  },
});

// console.log(summarization)
//----------------------//

const questionAnswer = await hf.questionAnswer({
  model: 'deepset/roberta-base-squad2',
  inputs: {
    question: 'What is the capital of France?',
    context: 'The capital of France is Paris.'
  }
})
// console.log(questionAnswer)

const tableQuestionAnswer = await hf.tableQuestionAnswer({
  model: 'google/tapas-base-finetuned-wtq',
  inputs: {
    query: '¿Cuál de las peliculas tiene una duración de 142 exactamente?',
    table: {
      Película: ['Harry Potter y la piedra filosofal', 'Harry Potter y la cámara secreta', 'Harry Potter y el prisionero de Azkaban', 'Harry Potter y el cáliz de fuego', 'Harry Potter y la orden del fénix', 'Harry Potter y el misterio del príncipe', 'Harry Potter y las reliquias de la muerte: parte 1', 'Harry Potter y las reliquias de la muerte: parte 2'],
      Duración: ['152', '161', '142', '157', '138', '153', '146', '130']
    }
  },
  options: {
    use_cache: true, // Valor por defecto: true.
    wait_for_model: false // Valor por defecto: false.
  }
})

/*console.log(tableQuestionAnswer.answer);
console.log(tableQuestionAnswer.coordinates);
console.log(tableQuestionAnswer.cells);
console.log(tableQuestionAnswer.aggregator);
*/


const textClassification = await hf.textClassification({
  model: 'nlptown/bert-base-multilingual-uncased-sentiment',
  inputs: 'Me encanta este producto, es excelente!',
  options: {
    use_cache: true, // Usa caché para acelerar las consultas
    wait_for_model: false // No esperes al modelo si no está listo
  }
});

//console.log(textClassification)

const textClassification2 = await hf.textClassification({
  model: 'distilbert-base-uncased-finetuned-sst-2-english',
  inputs: 'Me encanta este producto, es excelente!',
  options: {
    use_cache: true, // Whether to use cache or not
    wait_for_model: false // Whether to wait for the model to be ready
  }
});

// console.log(textClassification2)

const textGeneration = await hf.textGeneration({  // Tarda un poco en responder
  model: 'datificate/gpt2-small-spanish',
  inputs: 'El día estaba soleado y',
  parameters: {
    top_k: null, // Integer to define the top tokens considered within the sample operation to create new text.
    top_p: null, // Float to define the tokens that are within the sample operation of text generation.
    temperature: 1.0, // Float (0.0-100.0). The temperature of the sampling operation.
    repetition_penalty: null, // Float (0.0-100.0). The more a token is used within generation the more it is penalized to not be picked in successive generation passes.
    max_new_tokens: null, // Int (0-250). The amount of new tokens to be generated, this does not include the input length it is a estimate of the size of generated text you want.
    max_time: null, // Float (0-120.0). The amount of time in seconds that the query should take maximum.
    return_full_text: true, // Bool. If set to False, the return results will not contain the original query.
    num_return_sequences: 1, // Integer. The number of proposition you want to be returned.
    do_sample: true // Bool. Whether or not to use sampling, use greedy decoding otherwise.
  },
  options: {
    use_cache: true, // Whether to use cache or not
    wait_for_model: false // Whether to wait for the model to be ready
  }
});

// console.log(textGeneration);

const textGeneration2= await hf.textGeneration({  // Tarda un poco en responder
  model: 'gpt2',
  inputs: 'El día estaba soleado y',
  parameters: {
    top_k: null, // Integer to define the top tokens considered within the sample operation to create new text.
    top_p: null, // Float to define the tokens that are within the sample operation of text generation.
    temperature: 1.0, // Float (0.0-100.0). The temperature of the sampling operation.
    repetition_penalty: null, // Float (0.0-100.0). The more a token is used within generation the more it is penalized to not be picked in successive generation passes.
    max_new_tokens: null, // Int (0-250). The amount of new tokens to be generated, this does not include the input length it is a estimate of the size of generated text you want.
    max_time: null, // Float (0-120.0). The amount of time in seconds that the query should take maximum.
    return_full_text: true, // Bool. If set to False, the return results will not contain the original query.
    num_return_sequences: 1, // Integer. The number of proposition you want to be returned.
    do_sample: true // Bool. Whether or not to use sampling, use greedy decoding otherwise.
  },
  options: {
    use_cache: true, // Whether to use cache or not
    wait_for_model: false // Whether to wait for the model to be ready
  }
});

// console.log(textGeneration2);


const tokenClassification = await hf.tokenClassification({
  model: 'dbmdz/bert-large-cased-finetuned-conll03-english',
  inputs: 'Mi nombre es Sarah Jessica Parker pero puedes llamarme Jessica',
  parameters: {
    aggregation_strategy: 'simple',
  },
  options: {
    use_cache: true,
    wait_for_model: false,
  },
});

console.log(tokenClassification);

