## Escuela Colombiana de Ingeniería
### Arquitecturas Empresariales – AREP
# Laboratorio 9
Este proyecto introduce un sistema de búsqueda de documentos que hace uso de la biblioteca LangChain junto con el servicio de almacenamiento vectorial Pinecone. El propósito fundamental es capacitar la realización eficiente de búsquedas en un conjunto de documentos, aprovechando la capacidad de los modelos de lenguaje para codificar el texto en vectores semánticos.

## Elementos Necesarios
* **Phyton:** Es imprescindible contar con Python 3.7 o una versión posterior. Para instalar Python, visita el sitio web oficial: https://www.python.org/downloads/.
* **Pip:** Es el administrador de paquetes de Python, viene incluido con la instalación estándar de Python.
* **Phyton:**  Para crear y utilizar un entorno virtual de Python para gestionar las dependencias del proyecto de forma aislada puedes optar por venv o conda para crear y administrar estos entornos virtuales.
* **OpenAPI:** Para acceder a los servicios de embeddings de OpenAI, necesitarás una clave API válida. Regístrate para obtener una clave API en https://www.openai.com/.
* **Pinecode:**  Para aprovechar el servicio de almacenamiento vectorial de Pinecone, también necesitarás una clave API. Regístrate en https://www.pinecone.io/ para obtener tus claves de API.

## Pasos para la ejecucion
1. Clonamos el proyecto:
  ``` 
  git clone https://github.com/JuanFe2001/LAB9-AREP.git
  ```
2. Vamos a la carpeta del proyecto:
  ``` 
  cd LAB9-AREP
  ```
3. Instalamos las dependencias del proyecto:
  ``` 
  pip install -r librerias.txt
  ```
4. Configurar las variables de entorno en cada uno de los scripts con tus llaves:
  ``` 
  os.environ["OPENAI_API_KEY"] = <Tu OpenAI API Key>
  os.environ["PINECONE_API_KEY"] = <Tu Pinecone API Key>
  ```
5. Ejecutamos los Scripts
  ``` 
  python Punto1.py
  ```
  ``` 
  python Punto2.py
  ```
  ``` 
  python Punto3.py
  ```
## Demostracion:

**Punto1**

 ``` 
  from langchain.chains import LLMChain
#from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate

import os

os.environ["OPENAI_API_KEY"] = "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"

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = OpenAI()

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What is at the core of Popper's theory of science?"

response = llm_chain.run(question)
print(response)
 ```
1. Importaciones de bibliotecas:
Se importan las clases y funciones necesarias de las bibliotecas langchain y langchain_community para construir y ejecutar la cadena de lenguaje.
2. Configuración de la clave de API de OpenAI:
Se establece la clave de API de OpenAI como una variable de entorno. Esta clave es necesaria para acceder a los servicios de OpenAI, pero en este caso parece estar vacía.
3. Definición de la plantilla:
Se crea una plantilla que especifica el formato de las preguntas y respuestas. La variable {question} se utiliza para insertar la pregunta en la plantilla.
4. Inicialización del modelo de cadena de lenguaje:
Se crea una instancia de PromptTemplate utilizando la plantilla definida anteriormente, especificando que la plantilla contiene una variable llamada "question".
5.Creación del modelo de lenguaje:
Se crea una instancia del modelo de lenguaje de OpenAI utilizando la implementación proporcionada por langchain_community.
6. Creación de la cadena de lenguaje:
Se inicializa una cadena de lenguaje utilizando el modelo de lenguaje y la plantilla especificada.
7. Ejecución de la cadena de lenguaje con una pregunta específica:
Se define una pregunta específica.
Se ejecuta la cadena de lenguaje con la pregunta especificada.
La respuesta generada por la cadena de lenguaje se imprime en la consola.
![image](https://github.com/JuanFe2001/LAB9-AREP/assets/123691538/509356e4-5e13-441c-afcc-eca90bb09e6b)


