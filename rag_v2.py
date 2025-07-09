from langchain_core.runnables import  RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_ollama import OllamaEmbeddings
import os
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from neo4j import  Driver

from dotenv import load_dotenv

load_dotenv()

graph = Neo4jGraph(
    url="neo4j+s://ac78a829.databases.neo4j.io",     # or your actual Neo4j URL
    username="neo4j",
    password="jPG_oE4ySe25ZO2X2STL7vgKG8CYNQoW1UmKQ365Vag"         # replace with your real password
)

# llm = OllamaFunctions(model="llama3.1:latest",format="json", temperature=0)
llm = OllamaFunctions(model="llama3.1:latest", format="json", temperature=0, base_url="http://140.31.105.132:11434")

llm_transformer = LLMGraphTransformer(llm=llm)

driver = GraphDatabase.driver(
        uri = "neo4j+s://ac78a829.databases.neo4j.io",
        auth = ("neo4j",
                "jPG_oE4ySe25ZO2X2STL7vgKG8CYNQoW1UmKQ365Vag"))

def create_fulltext_index(tx):
    query = '''
    CREATE FULLTEXT INDEX `fulltext_entity_id` 
    FOR (n:__Entity__) 
    ON EACH [n.id];
    '''
    tx.run(query)

# Function to execute the query
def create_index():
    with driver.session() as session:
        session.execute_write(create_fulltext_index)
        print("Fulltext index created successfully.")

# Call the function to create the index
try:
    create_index()
except:
    pass

# Close the driver connection
# driver.close()

#create entities or the subj on your query e.g "who is maria?" possible entity is maria then 'Maria' will be search thru the graph DB
class Entities(BaseModel):
    """Identifying information about entities."""

    names: list[str] = Field(
        ...,
        description=""" Extract all the possible subject in the text""",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization, person or the subject as entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)
# llm_trans= OllamaFunctions(model="mistral", temperature=0, format="json")
llm_trans = OllamaFunctions(model="llama3.1:latest", format="json", temperature=0, base_url="http://140.31.105.132:11434")
entity_chain = llm_trans.with_structured_output(Entities)

def graph_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke(question)
    print(entities)
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
                YIELD node, score
                CALL {
                WITH node
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                RETURN output LIMIT 50
            """,
            {"query": entity},
        )
        result += "\n".join([el['output'] for el in response])
    return result


def full_retriever(question: str):
    print(question)
    graph_data = graph_retriever(question)
    print(graph_data)
    # docs_with_score = db.similarity_search_with_score(question, k=5)
    # page_contents = [doc[0].page_content for doc in docs_with_score]
    # print(page_contents)

    final_data = f"""Relationships:
{graph_data}

    """
    return final_data


# combination of embedding and graph search
template = """
Answer the question based only on the following context:
{context}

You are an advanced AI designed to analyze and synthesize information from a single provided file: {context} containing both factual details and relationship connections about a subject. 
When answering a question, do not simply extract text but instead interpret and expand upon the provided information by logically inferring connections and implications. 
Identify key facts, analyze relationships, and generate well-structured responses that go beyond surface-level details while maintaining accuracy and coherence. 
Use contextual reasoning to provide insightful and relevant answers. If the required information is not found, acknowledge the limitation while avoiding speculation. 
Always maintain a neutral, well-supported, and logically sound tone in your responses.

when answering a question be confident and make it so like you're the one answering the question based from your knowldege not from external source 
NEVER mention about the data source or where you get the data
when answering a question, answer it fully 

Question: {question}
Use natural human language 


Answer:"""
# llm2 = OllamaFunctions(model="mistral", temperature=1, format="json")
llm2 = OllamaFunctions(model="llama3.1:latest", format="json", temperature=0, base_url="http://140.31.105.132:11434")

prompt = ChatPromptTemplate.from_template(template)

chain = (
        {   
            "context": full_retriever,
            "question": RunnablePassthrough(),
        }
    | prompt
    | llm2
    | StrOutputParser()
)

#ask a question
# print(chain.invoke(input="who is sophia?"))
# chain.invoke(input="what is nist?")

if __name__ == "__main__":
    print("🔍 Retrieval-Augmented Generation (RAG) via Ollama + Neo4j")
    while True:
        try:
            question = input("\nAsk a question (or 'exit'): ").strip()
            if question.lower() in ["exit", "quit"]:
                break
            print(chain.invoke(input=question))
        except Exception as e:
            print(f"❌ Error: {e}")
