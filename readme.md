# Huggingface.js HfInference

Instalación para su funcionamiento inicial; si ya está instalado en el proyecto, solo usar npm install.
> npm install @huggingface/inference

<sub>Este código importa la biblioteca **@huggingface/inference** que proporciona una interfaz para usar modelos de lenguaje y de aprendizaje automático pre-entrenados en la nube a través de la plataforma Hugging Face. La biblioteca incluye funciones para diversas tareas de procesamiento de lenguaje natural (PLN), así como para tareas de visión por computadora y procesamiento de audio.</sub>

> API REFERENCIA: [huggingface.js](https://huggingface.co/docs/huggingface.js/inference)

El codigo usa `Natural Language`

#

### **fillMask**

> esta función utiliza el modelo de lenguaje BERT para rellenar el espacio en blanco en una oración dada.

Parametros:
- **`model (obligatorio)`**: el nombre o la ruta del modelo de lenguaje pre-entrenado que se utilizará para rellenar los espacios en blanco. Ejemplo: `'bert-base-uncased'`
- **`inputs (obligatorio)`**: la oración que se va a completar, con los espacios en blanco representados como `[MASK]`. Ejemplo: `'[MASK] Mundo!'`
- **`top_k (opcional)`**: un entero que indica el número de opciones de relleno a considerar. Por defecto, se consideran las 5 opciones más probables. Ejemplo: `10`
- **`use_cache (opcional)`**: un booleano que indica si se deben almacenar en caché los resultados de las llamadas anteriores a la función. Por defecto, es `false`. Ejemplo: `true`

Ejemplo:
```typescript
const fillMask = await hf.fillMask({
  model: 'bert-base-uncased',
  inputs: 'La [MASK] de París es conocida como la Ciudad de la Luz.',
  top_k: 3,
  use_cache: false
})
console.log(fillMask);
```
> <sub>Este ejemplo utilizaría el modelo `bert-base-uncased` para completar la oración `'La [MASK] de París es conocida como la Ciudad de la Luz.'`, considerando solo las 3 opciones de relleno más probables y sin almacenar los resultados en caché. El resultado sería un objeto que contiene una matriz de objetos con las opciones de relleno y sus respectivas probabilidades.</sub>

#

### **summarization**

> esta función utiliza el modelo de lenguaje BART para generar un resumen de un texto dado.

Parametros:
- **`model (obligatorio)`**: el modelo a utilizar para el resumen del texto. Debe ser una cadena de texto que identifique el modelo, como por ejemplo `'facebook/bart-large-cnn'`..
- **`inputs (obligatorio)`**: el texto a resumir. Debe ser una cadena de texto que contenga el texto completo que se desea resumir.
- **`parameters (opcional)`**: un objeto que contiene los parámetros del modelo que se desean cambiar. Por ejemplo, si se desea cambiar el número máximo de palabras en el resumen, se puede establecer el parámetro `max_length`. El valor predeterminado para este parámetro es None, lo que significa que se utilizarán los valores predeterminados del modelo. Otros parámetros comunes incluyen `min_length`, `num_beams`, `length_penalty`, `no_repeat_ngram_size`, `early_stopping`, entre otros.
- `use_cache (opcional)`: un valor booleano que indica si se debe utilizar el caché para la inferencia. El valor predeterminado es `true`.

Ejemplo:
```typescript
const summarization = await hf.summarization({
  model: "facebook/bart-large-cnn",
  inputs:
    "El huracán Katrina fue uno de los huracanes más destructivos en la historia de los Estados Unidos, causando daños catastróficos en Luisiana y Misisipi en 2005. Las inundaciones resultantes y los fuertes vientos causaron daños en la infraestructura, incluidas las carreteras y los puentes. También causó daños en los sistemas de energía eléctrica y de agua, y dejó a millones de personas sin hogar.",
  parameters: {
    max_length: 50,
    num_beams: 2,
    length_penalty: 1.5,
  },
  use_cache: true,
});
console.log(summarization);
```
> <sub>En este ejemplo, se está utilizando el modelo `'facebook/bart-large-cnn'` para generar un resumen del texto proporcionado sobre el huracán Katrina. Además, se está estableciendo el parámetro `max_length` en 50, lo que significa que el resumen generado no tendrá más de 50 palabras. También se están utilizando 2 beams y un `length_penalty` de 1.5 para ajustar el equilibrio entre la coherencia y la concisión del resumen. Por último, se está utilizando el caché para acelerar la inferencia. El resultado del resumen generado se imprimirá en la consola con `console.log(summarization)`.</sub>

#

### **questionAnswer**

> esta función utiliza el modelo de lenguaje RoBERTa para responder preguntas en un contexto dado

Parametros:
- **`model (obligatorio)`**: el modelo a utilizar para responder la pregunta. Debe ser una cadena de texto que identifique el modelo, como por ejemplo `'deepset/roberta-base-squad2'`
- **`inputs (obligatorio)`**: un objeto con dos claves: `question` y `context`. La clave `question` debe contener la pregunta a responder, mientras que la clave `context` debe contener el texto en el que se encuentra la respuesta. Ambos deben ser cadenas de texto.
- **`parameters (opcional)`**: un objeto que contiene parámetros adicionales para la función. En el caso de questionAnswer, los siguientes parámetros están disponibles.
  - `max_answer_length`: La longitud máxima permitida para la respuesta. Por defecto es 15. Ejemplo: `{max_answer_length: 50}`.

Ejemplo:
```typescript
const questionAnswer = await hf.questionAnswer({
  model: "deepset/roberta-base-squad2",
  inputs: {
    question: "¿Cuál es el nombre del presidente de los Estados Unidos?",
    context:
      "El presidente de los Estados Unidos es Joe Biden. Joe Biden es el 46º presidente de los Estados Unidos.",
  },
  parameters: {
    max_answer_length: 50,
  },
});
console.log(questionAnswer);
```

https://huggingface.co/docs/api-inference/detailed_parameters
https://huggingface.co/docs/huggingface.js/main/en/inference/README
https://huggingface.co/openai/clip-vit-large-patch14
