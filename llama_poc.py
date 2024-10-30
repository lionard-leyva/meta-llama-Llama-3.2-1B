import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Obtenemos el token desde la variable de entorno
# os.getenv busca el valor de una variable de entorno llamada "HUGGINGFACE_TOKEN".
# Si encuentra esta variable, devuelve su valor; si no la encuentra, lanza un error.
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Definimos el modelo que queremos usar
# "MODEL_ID" contiene el identificador del modelo en Hugging Face.
MODEL_ID = "meta-llama/Llama-3.2-1B"

# Verificamos si el token está presente; si no, se lanza un error.
if HUGGINGFACE_TOKEN is None:
    raise ValueError("El token de Hugging Face no se encuentra en las variables de entorno. Por favor, configura HUGGINGFACE_TOKEN.")

# Cargamos el tokenizador desde el modelo preentrenado especificado.
# Utilizamos el token para autenticar la descarga del modelo desde Hugging Face.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HUGGINGFACE_TOKEN)

# Cargamos el modelo desde Hugging Face utilizando el token para autenticarnos.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HUGGINGFACE_TOKEN)

# Definimos el prompt que le daremos al modelo para generar texto.
# Este es el mensaje de entrada que el modelo usará para generar una respuesta.
prompt = "Por favor responde en español Explícale en español a un niño de 5 años qué es la programación. Usa un lenguaje sencillo y da ejemplos fáciles de entender."

# Tokenizamos el prompt para convertirlo en una representación numérica.
# "return_tensors='pt'" indica que queremos los tensores en el formato de PyTorch.
inputs = tokenizer(prompt, return_tensors="pt")

# Generamos la respuesta del modelo a partir de los tokens de entrada.
# max_length define cuántas palabras máximas debe tener la respuesta.
# do_sample=True indica que la generación debe ser aleatoria, no siempre la misma.
# temperature=0.7 controla cuánta aleatoriedad hay en la respuesta; valores más altos hacen que sea más creativo.
# repetition_penalty penaliza la repetición para evitar que el modelo repita constantemente.
# top_k y top_p ayudan a limitar las opciones para evitar respuestas repetitivas.
outputs = model.generate(
    **inputs,
    max_length=200,
    do_sample=True,
    temperature=0.7,
    repetition_penalty=1.2,
    top_k=50,
    top_p=0.9
)

# Decodificamos la respuesta del modelo para convertirla de nuevo en texto.
# "skip_special_tokens=True" elimina tokens especiales que no son parte del texto natural.
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Imprimimos el texto generado por el modelo.
print(generated_text)
