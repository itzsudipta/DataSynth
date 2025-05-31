from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnableSequence
import gradio as gr

# Load model
model_id = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7
)

# LangChain wrapper
llm = HuggingFacePipeline(pipeline=generator)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# Runnable sequence instead of LLMChain
chain = prompt | llm

# Gradio interface
def generate_answer(question):
    result = chain.invoke({"question": question})
    return result

gr.Interface(fn=generate_answer, inputs="text", outputs="text", title="Gemma 2B Chat").launch()