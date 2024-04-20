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

Este código utiliza la biblioteca langchain para crear un modelo de cadena de lenguaje que responde a preguntas basadas en una plantilla específica
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

**Punto2**

```
import bs4
from langchain import hub
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os

os.environ["OPENAI_API_KEY"] = "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"


loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(splits[0])
print(splits[1])

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is Task Decomposition?")

print(response)
```
Este código realiza varias tareas relacionadas con el procesamiento de lenguaje natural (NLP) y la generación de texto utilizando modelos avanzados de inteligencia artificial. 

1. Importa las bibliotecas necesarias, incluyendo bs4 (Beautiful Soup), que es una biblioteca de Python para extraer datos de archivos HTML y XML, y varios módulos de langchain_community relacionados con el procesamiento de lenguaje natural y la generación de texto.
2. Configura una clave de API de OpenAI para acceder a sus servicios.
3.Carga documentos de la web a través de WebBaseLoader. En este caso, carga el contenido de la página "https://lilianweng.github.io/posts/2023-06-23-agent/".
4. Divide los documentos cargados en fragmentos más pequeños utilizando RecursiveCharacterTextSplitter. Esto puede ayudar en el procesamiento posterior y la generación de texto.
5. Crea un "vectorstore" utilizando Chroma y OpenAIEmbeddings para representar los fragmentos de texto en un espacio vectorial.
6. Configura un modelo de recuperación utilizando el vectorstore creado anteriormente.
7. Configura un modelo de lenguaje basado en RAG (Retrieval-Augmented Generation) utilizando el modelo "rlm/rag-prompt" de la biblioteca hub.
8. Configura un modelo de lenguaje para la conversación utilizando el modelo "gpt-3.5-turbo" de OpenAI.
9. Define una función para formatear los documentos cargados.
Configura una cadena de procesamiento para la generación de texto. Esta cadena incluye el modelo de recuperación, el modelo RAG y el modelo de lenguaje para la conversación.
10. Invoca la cadena de procesamiento para generar una respuesta a la pregunta "What is Task Decomposition?".
11. Imprime la respuesta generada.
![image](https://github.com/JuanFe2001/LAB9-AREP/assets/123691538/8215e51d-7766-461a-b18c-9277c47861e0)

**Punto3**
```
from langchain_community.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, PodSpec
import os

os.environ["OPENAI_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"] = "58e1668f-5c14-4af5-8b08-ac18efd1f81d"
os.environ["PINECONE_ENV"] = "gcp-starter"

def loadText():
    loader = TextLoader("Conocimiento.txt")
    documents = loader.load()
    #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
        is_separator_regex = False,
    )


    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    import pinecone


    index_name = "langchain-demo"
    pc = Pinecone(api_key='eb0f1c59-78f7-4e47-9017-87941c145474')

    print(pc.list_indexes())

    # First, check if our index already exists. If it doesn't, we create it
    if len(pc.list_indexes())==0:
        # we create a new index
        #pc.create_index(name=index_name, metric="cosine", dimension=1536)
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=PodSpec(
                environment=os.getenv("PINECONE_ENV"),
                pod_type="p1.x1",
                pods=1
            )
        )

    # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
    docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

def search():
    embeddings = OpenAIEmbeddings()

    index_name = "langchain-demo"
    # if you already have an index, you can load it like this
    docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

    query = "What is a saxophone?"
    docs = docsearch.similarity_search(query)

    print(docs[0].page_content)

loadText()
search()
```
Este código realiza varias operaciones relacionadas con la carga y búsqueda de texto utilizando diferentes servicios y bibliotecas.

1. Importaciones de bibliotecas:
Se importan varias clases y funciones de diferentes bibliotecas necesarias para realizar las operaciones de carga y búsqueda de texto.
2. Configuración de las claves de API y entorno:
Se establecen las claves de API necesarias para acceder a los servicios de OpenAI y Pinecone, así como el entorno de Pinecone.
3. Función loadText():
Se define una función llamada loadText() que se encarga de cargar el texto desde un archivo llamado "Conocimiento.txt".
Se utiliza un TextLoader para cargar los documentos del archivo.
Se define un RecursiveCharacterTextSplitter para dividir los documentos en fragmentos más pequeños.
Se crea un índice en Pinecone utilizando los documentos divididos y los embeddings de OpenAI.
La función imprime los índices disponibles antes de crear uno nuevo y luego carga los documentos en Pinecone.
4. Función search():
Se define una función llamada search() que se encarga de realizar la búsqueda de similitud de documentos.
Se utiliza un índice existente en Pinecone y los embeddings de OpenAI para realizar la búsqueda.
Se proporciona una consulta (query) para buscar documentos similares.
La función imprime el contenido de los documentos más similares encontrados.
5. Ejecución del código:
Se llama a la función loadText() para cargar los documentos y crear un índice en Pinecone.
Se llama a la función search() para realizar una búsqueda de similitud de documentos.

## Autor:
* Juan Felipe Vivas Manrique


