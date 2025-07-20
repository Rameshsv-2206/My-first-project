import pandas as pd

df = pd.read_csv('sales_data.csv')


import subprocess
subprocess.check_call(["pip", "install", "-U", "langchain-community"])

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

os.listdir('PDF Folder')

pdf_folder = 'PDF Folder'
documents = []
for file in os.listdir(pdf_folder):
    if file.endswith('.pdf'):
        loader = PyPDFLoader(os.path.join(pdf_folder, file))
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

print(f"Total number of text chunks: {len(texts)}") 
print("First chunk example:")
print(texts[1].page_content[:500] + "...")

import pickle
with open('processed_texts.pkl', 'wb') as f:
    pickle.dump(texts, f)
print("Processed texts saved to 'processed_texts.pkl'")

import numpy as np
from scipy import stats

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain

chat_model = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")




def create_advanced_summary(df):
    
    if not isinstance(df, pd.DataFrame):
      return ""
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')  
    
    total_sales = df['Sales'].sum()
    avg_sale = df['Sales'].mean()
    median_sale = df['Sales'].median()  # Corrected: Use .median() instead of .mediap()
    sales_std = df['Sales'].std()

    df['Month'] = df['Date'].dt.to_period('M')
    monthly_sales = df.groupby('Month', observed=False)['Sales'].sum().sort_values(ascending=False)
    best_month = str(monthly_sales.index[0])
    worst_month = str(monthly_sales.index[-1])
    
    product_sales = df.groupby('Product', observed=False)['Sales'].agg(['sum', 'count', 'mean'])
    top_product = product_sales['sum'].idxmax()
    most_sold_product = product_sales['count'].idxmax()

    region_sales = df.groupby('Region', observed=False)['Sales'].sum().sort_values(ascending=False)
    best_region = region_sales.index[0]
    worst_region = region_sales.index[-1]

    avg_satisfaction = df['Customer_Satisfaction'].mean()  # Corrected: Use 'Customer_Satisfaction' column name
    satisfaction_std = df['Customer_Satisfaction'].std()

    age_bins = [0, 25, 35, 45, 55, 100]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '55+']
    df['Age_Group'] = pd.cut(df['Customer_Age'], bins=age_bins, labels=age_labels, right=False)
    age_group_sales = df.groupby('Age_Group', observed=False)['Sales'].mean().sort_values(ascending=False)
    best_age_group = age_group_sales.idxmax()

    gender_sales = df.groupby('Customer_Gender')['Sales'].mean()
    
    summary = f"""
    
    - Total Sales: ${total_sales}
    - Average Sale: ${avg_sale:.2f}
    - Median Sale: ${median_sale}
    - Sales Standard Deviation: ${sales_std:.2f}
    f#"- Sales Standard Deviation: ${sales_std:.2f}"
    
    - Best Performing Month: {best_month}
    - Worst Performing Month: {worst_month}
    
    - Top Selling Product (by value): {top_product}
    - Most Frequently Sold Product: {most_sold_product}
    
    - Best Performing Region: {best_region}
    - Worst Performing Region: {worst_region}
    
    - Average Customer Satisfaction: {avg_satisfaction: .2f}/5
    - Customer Satisfaction Standard Deviation: {satisfaction_std: .2f}
    - Best Performing Age Group: {best_age_group}"
    - Gender-based Average Sales: 
        - Male: ${gender_sales.get('Male', 0):.2f}
        - Female: ${gender_sales.get('Female', 0):.2f}
    1. The sales data shows significant variability with a standard deviation of ${sales_std:.2f}."
    2. The {best_age_group} age group shows the highest average sales."
    3. Regional performance varies significantly, with {best_region} outperforming {worst_region}."
    4. The most valuable product ({top_product}) differs from the most frequently sold product ({most_sold_product})."
    """
    

    return summary
 

advanced_summary = create_advanced_summary(df)

#pip install -U langchain-openai
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

chain = prompt | llm
response = chain.invoke({"product": "custom sneakers"})

print(response.content)



scenario_template = """
You are an expert sales analyst. Use the foillowing advanced sales data to provide in-depth sales insights and actionable recommendations.
Be specfific and refer to the data points provided.

{advanced_summary}

Question = {question}

Detailed analysis and recommendations:
"""


prompt = PromptTemplate(template=scenario_template, input_variables=["advanced_summary", "question"])
llm_chain = LLMChain(prompt=prompt, llm=chat_model)

def generate_insight(advanced_summary, question):
    return llm_chain.run(advanced_summary=advanced_summary, question=question)

question = "Based on this data, what are our main areas of improvement and what startegies would you recommend to boost sales and customer satisfaction?"
insight = generate_insight(advanced_summary, question)
print(insight)

from langchain.chains import SequentialChain

data_analysis_template = """
Analyze the following advanced sales data summary:

{advanced_summary}

Provide a concise analysis of the key points:
"""
data_analysis_prompt = PromptTemplate(template="Analyze this data: {advanced_summary}", input_variables=["advanced_summary"])
data_analysis_chain = LLMChain(llm=chat_model, prompt=data_analysis_prompt, output_key="analysis")

recommendation_template = """
Based on the following analysis of sales data:
    
{analysis}

Provide specific recommendations to address the question: {question}

Recommendations:
"""

recommendation_prompt = PromptTemplate(template=recommendation_template, input_variables=["analysis", "question"])
recommendation_chain = LLMChain(llm=chat_model, prompt=recommendation_prompt, output_key="recommendations")


overall_chain = SequentialChain(
    chains= [data_analysis_chain, recommendation_chain], 
    input_variables = ["advanced_summary", "question"], 
    output_variables = ["analysis", "recommendations"], 
    verbose=True
)
    
def generate_chained_insight(question):
    try:
        result = overall_chain({"advanced_summary": advanced_summary, "question": question}) 
        return f"Analysis:\n{result['analysis']}\n\nRecommendations:\n{result['recommendations']}" 
    except Exception as e:
        print(f"Error in generate_chained_insight: {e}")
        return f"Error: {str(e)}"


test_question = "How can we improve sales in our worst-performing region?"
chained_insight = generate_chained_insight(test_question)
print (f"Question: {test_question}")
print (f"Chained Insight:\n{chained_insight}")


from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.utilities import WikipediaAPIWrapper 
from datetime import datetime, timedelta

import pickle
with open('processed_texts.pkl', 'rb') as f:
 texts = pickle.load(f)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

retriever = vectorstore.as_retriever(search_types="similarity", search_kwrgs={"k":3})

chat_model = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import Tool 
import wikipedia
from bs4 import BeautifulSoup

wikipedia_wrapper = WikipediaAPIWrapper()

def wiki_search(query):
    try:
        content = wikipedia_wrapper.run(query)
    
        search_results = wikipedia.search(query, results=3) 
        urls = []
        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                soup = BeautifulSoup (page.html(), features="lxml")
                content += f"\nTitle: {page.title}\nSummary: {soup.get_text()[:500]}...\n" 
                urls.append(page.url)
            except (wikipedia.exceptions. DisambiguationError, wikipedia.exceptions.PageError):
                continue
        return {'content': content, 'urls': urls}
    except Exception as e:
        return {'content': f"An error occurred: {str(e)}", 'urls': []}

wikipedia_tool = Tool(
    name="Wikipedia Search",
    func=wiki_search,
    description="Searches Wikipedia for information"
)

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

chat_model = ChatOpenAI(
    temperature=0.7,  # Adjusts creativity of responses
    model_name="gpt-3.5-turbo"  # You can use "gpt-4" if available
)

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=chat_model,
    memory=memory,
    verbose=True
)

def generate_rag_insight_with_memory_seqchain (question):
    retrieved_documents = retriever.get_relevant_documents(question)
    retrieved_texts = " ".join([doc.page_content for doc in retrieved_documents])

    context = {
    "advanced_summary": advanced_summary,
    "question": question
    }
    analysis_result = overall_chain.apply([context])[0] 
    analysis = analysis_result['analysis']
    recommendations = analysis_result['recommendations']

    wiki_results = wikipedia_tool.run(question) 
    wiki_content = wiki_results['content']
    wiki_urls = wiki_results['urls']

    enhanced_context = f"{analysis}\n\nAdditional information from Wikipedia: \n{wiki_content}"

    final_result = conversation.predict(input=enhanced_context) 
    insight = final_result
    sources = [doc.metadata['source'] for doc in retrieved_documents]

    sources.extend([f"wikipedia: {url}" for url in wiki_urls])
    return f"Analysis: \n{analysis}\n\nRecommendations: \n{recommendations }\n\nEnhanced Insight: \n{insight}\n\nSources: \n" + "\n".join(set (sources))


def generate_rag_insight (question):
    context = f"Advanced Sales Summary: \n{advanced_summary}\n\nQuestion: {question}" 
    result = qa_chain ({"query": context})
    
    wiki_results = wikipedia_tool.run(question)
    wiki_content = wiki_results['content']
    wiki_urls = wiki_results['urls']
    
    enhanced_context = f"{context}\n\nAdditional information from Wikipedia: \n{wiki_content}"
    
    final_result = qa_chain ({"query": enhanced_context})
    insight = final_result['result']
    sources = [doc.metadata['source'] for doc in final_result['source_documents']]
    
    sources.extend( [f"Wikipedia: {url}" for url in wiki_urls])
    return f"Insight: {insight}\n\nSources: \n" + "\n".join(set (sources))


def generate_insight_with_memory(question):
    return conversation.predict(input=f"Advanced sales Summary:\n{advanced_summary}\n\nQuestion: {question}")


print(generate_insight_with_memory("What are our top-selling products?"))

test_question = "How can we improve customer satisfaction bsed on our sales trends?"
print(generate_rag_insight(test_question))

import subprocess
subprocess.check_call(["pip", "install", "-U", "langchain-community"])

from langchain.chat_models import ChatOpenAI  # Import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import StringPromptTemplate
import matplotlib.pyplot as plt
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.evaluation.qa import QAEvalChain
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool  # Import Tool
from langchain.agents import ZeroShotAgent
from typing import List, Union
import pandas as pd

df = pd.DataFrame({
    "Product": ["A", "B", "C", "A", "B"],
    "Sales": [100, 200, 150, 120, 250],
    "Date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
    "Customer_Satisfaction": [4.5, 3.8, 4.2, 4.7, 3.9]
})

def plot_product_category_sales():
    product_cat_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    product_cat_sales.plot(kind='bar')
    plt.title('Sales Distribution by Product')
    plt.xlabel('Product')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.tight_layout() 


def plot_sales_trend():
    plt.figure(figsize=(10, 6))
    df.groupby('Date') ['Sales'].sum().plot()
    plt.title('Daily Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')


def advanced_summary(x):
    return "This is an advanced summary of sales data."

def generate_rag_insight(x):
    return "This is a generated insight using RAG system."

from langchain.tools import Tool  # Import Tool

tools = [
    Tool(
        name="ProductCategorySalesPlot",
        func=lambda x: plot_product_category_sales,
        description="Generates a plot of sales distribution by product category"
    ),
    Tool(
        name="SalesTrendPlot",
        func=lambda x: plot_sales_trend,
        description="Generates a plot of the daily sales trend"
    ),
    Tool(
        name="AdvancedSummary",
        func=lambda x: advanced_summary,
        description="Provides the advanced summary of sales data"
    ),
    Tool(
        name="RAGInsight",
        func=lambda x: generate_rag_insight,
        description="Generates insights using RAG system"
    )
]    

prefix = """You are an AI sales analyst with access to advanced sales data and a RAG system. Use the following tools to answer the user's questions: """
suffix = """Begin!"
{chat_history} Human: {input}
AI: Let's approach this step-by-step:
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
tools,
prefix=prefix,
suffix=suffix,
input_variables=["input", "chat_history", "agent_scratchpad"]
)



chat_model = ChatOpenAI(model="gpt-4", temperature=0)
llm_chain = LLMChain(prompt=prompt, llm=chat_model)

llm_chain = LLMChain(prompt=prompt, llm=chat_model)
agent = ZeroShotAgent(llm_chain=llm_chain, tools = tools, verbose = True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

response = agent_chain.run(input="Analyze our sales performance and suggest strategies for improvement. Include visualizations in your analysis.", chat_history="", agent_scratchpad="") 
print("Full agent response:")
print(response)

qa_pairs = [
        {
            "question": "What is our total sales amount?",
            "answer": f"The total sales amount is ${df['Sales'].sum():,.2f}."
        },
        {
            "question": "Which product category has the highest sales?",
            "answer": f"The product category with the highest sales is {df.groupby('Product')['Sales'].sum().idxmax()}."
        },
        {
            "question": "What is our average customer satisfaction score?",
            "answer": f"The average customer satisfaction score is {df['Customer_Satisfaction'].mean():.2f}."
        },
    ]

from langchain.evaluation.qa import QAEvalChain
from langchain.chat_models import ChatOpenAI
print(dir(QAEvalChain))

chat_model = ChatOpenAI(model_name="gpt-4")

def evaluate_model(qa_pairs, agent_chain):
    eval_chain = QAEvalChain.from_llm (llm=chat_model, handle_parsing_errors=True)
    predictions = []
    for q in qa_pairs:
        result = agent_chain.run(input=q["question"], chat_history="", agent_scratchpad="") 
        predictions.append({"question": q["question"], "prediction": result})
    results = eval_chain.evaluate(
        examples=qa_pairs,
        predictions=predictions,
        question_key="question",
        answer_key="answer",
        prediction_key="prediction"
    )

    eval_results = []
    results = eval_chain.evaluate(
        examples=qa_pairs,
        predictions=predictions, 
        question_key="question", 
        answer_key="answer",
        prediction_key="prediction" # Ensure this matches the keys used in predictions
    )

    for i, result in enumerate(results):
        eval_results.append({
            "question": qa_pairs[i]["question"],
            "predicted": predictions[i]["prediction"], 
            "actual": qa_pairs[i]["answer"],
            "correct": result["results"] == 'result'
        })
    return eval_results

eval_results = evaluate_model(qa_pairs, agent_chain)
print("Evaluation Results:")
print(eval_results)

print("Model Evaluation results:")
for result in eval_results:
    print(f"Question: {result['question']}")
    print(f"Predicted: {result['predicted']}")
    print(f"Actual: {result['actual']}")
    print(f"Correct: {result['correct']}")
    print("---")      

accuracy = sum(1 for r in eval_results if r['correct']) / len(eval_results)
print(f"Model accuracy = {accuracy: 0.2%}")
    

import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

class SimpleModelMonitor:
    def __init__(self, log_file= 'simple_model_monitoring.json'): 
        self.log_file = log_file 
        self.logs = self.load_logs()
    def load_logs (self):
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f: 
                return json.load(f) 
        return []
        
    def log_interaction(self, query, execution_time):
        log_entry = {
            'timestamp' : datetime.now().isoformat(),
            'query': query,
            'execution_time': execution_time
        
        }
        self.logs.append(log_entry)
        self.save_logs()
    
    def save_logs(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.logs,f, indent=2)
            
    def plot_execution_times(self):
        timestamps = [datetime.fromisoformat(log['timestamp']) for log in self.logs]
        execution_times = [log['execution_time'] for log in self.logs]
        
        plt.figure(figsize = (10,5))
        plt.plot(timestamps, execution_times, marker = 'o')
        plt.title('Model Execution Times')
        plt.xlabel("Timestamp")
        plt.ylabel('Execution Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('simple_execution_times.png')
        plt.close()
        
    def get_average_execution_time(self):
        return np.mean([log['execution_time'] for log in self.logs])

model_monitor = SimpleModelMonitor()

import datetime # Simulate execution time
import matplotlib.pyplot as plt  # Required for plotting

class ModelMonitor:
    def __init__(self):
        self.logs = []

    def log_interaction(self, query, execution_time):
        self.logs.append({"query": query, "execution_time": execution_time})
        print(f"[DEBUGLOGGED] {query} - {execution_time:.2f} seconds")
        
    def get_average_execution_time(self):
        if not self.logs:
            return 0.0
        total_time = sum(log["execution_time"] for log in self.logs)
        return total_time / len(self.logs)
    
    def plot_execution_times(self):
        if not self.logs:
            print("No logs to plot.")
            return
        queries = [log["query"] for log in self.logs]
        times = [log["execution_time"] for log in self.logs]

        plt.figure(figsize=(10, 5))
        plt.barh(queries, times, color='skyblue')
        plt.xlabel('Execution Time (seconds)')
        plt.title('Query Execution Times')
        plt.tight_layout()
        plt.show()
    
model_monitor = ModelMonitor()  
    
def run_agent_with_monitoring(query):
    start_time = datetime.datetime.now()
    response = f"Simulated response for: {query}"
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    model_monitor.log_interaction(query, execution_time)
    return response, execution_time
    

test_queries = [
    "What are our top-selling products?",
    "How can we improve sales in our worst-performing region?",
    "What is the relationship between customer satisfaction and sales?"
]

for query in test_queries:
    response, execution_time = run_agent_with_monitoring(query)
    print(f"Query: {query}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print("---")
    
model_monitor.plot_execution_times()
avg_execution_time = model_monitor.get_average_execution_time()
print(f"Average Execution Time: {avg_execution_time:.4f} seconds")

model_monitor.plot_execution_times()

avg_execution_time = model_monitor.get_average_execution_time()
print(f"Average Execution Time: {avg_execution_time:.2f} seconds")

import streamlit as st


st.title("InsightForge: BusinessIntelligenceAssistant: robot_face: :bulb")
st.write("Use the sidebar to navigate through different sections of the application.")

page = st.sidebar.selectbox("Select a Page", ["Home", "Data Analysis", "AIAssistant", "Model Performance"])

if page == "Home":
    st.title("Home")
    
    st.header("Welcome to InsightForge")
    st.write("This application provides business intelligence insights.")
    pass

elif page == "Data Analysis":
    st.header("Data Analysis")   
    st.subheader("Sales Summary")
        
    try:
        st.write(advanced_summary)
    except NameError:
        st.write("Error: `advanced_summary` is not defined.")

    st.subheader("Sales Distribution by Product Category")
    try:
        fig_category = plot_product_category_sales()
        st.pyplot(fig_category)
    except NameError:
        st.write("Error: `plot_product_category_sales()` function is not defined.")
        
    st.subheader("Daily Sales Trend")
    try:
        fig_category = plot_sales_trend()
        st.pyplot(fig_category)
    except NameError:
        st.write("Error: `plot_sales_trend()` function is not defined.")
    pass    
        
elif page == "AIAssistant":
    st.header("AI Sales Analyst")
    st.write("This is where the AI Assistant functionality will go.")
    
    
    ai_mode = st.radio("Choose AI MOde:", ["Standard", "RAG Insights"])
    
    user_input = st.text_input("Ask a question about the sales data:")
    if user_input:
        if ai_mode=="Standard":
            start_time = datetime.now()
            response = agent_chain.run(input=user_input, chat_history="", agent_scratchpad="")
            end_time = datetime.now()
            execution_time = (end_time-start_time).total_seconds()
            
            st.write("AI Response:")
            st.write(response)
            
            model_monitor.log_interaction(user_input, execution_time)
            stwrite(f"Execution time: {execution_time:.2f} seconds")
            
        else: # RAG Insights mode
            start_time = datetime.now()
            rag_response = generate_rag_insight(user_input)
            endtime = datetime.now()
            execution_time = (end_time-start_time).total_seconds()
                               
            st.write("RAG Insight:")
            st.write(rag_response)
            
            model_monitor.log_interaction(user_input, execution_time)
            st.write(f"Execution time: {execution_time:.2f} seconds")   
            pass
            
elif page == "Model Performance":
    st.header("Model Performance")
    st.write("This is where the model performance metrics will be displayed.")
    
    st.subheader("Model Evaluation") 
    if st.button("Run Model Evaluation"):
        qa_pairs = [
        {
            "question" : "What is our total sales amount?",
            "answer" : f"The total sales amount is ${df['sales'].sum():,.2f}."
        },
        {
            "question" : "What product category has the highest sales",
            "answer" : f"The product category with the highest sales is {df.groupby('Product')['sales'].sum():,idxmax()}.",
        },
        {
            "question" : "What is our average customer satisfaction score",
            "answer" : f"The average customer satisfaction score is {df['customer_satisfaction'].mean():.2f}."
        }
    ]       
        
        eval_results = evaluate_model(qa_pairs)
        for result in eval_results:
            st.write(f"Question: {result['question']}")
            st.write(f"Predicted: {result['predicted']}")
            st.write(f"Actual: {result['actual']}")
            st.write(f"Correct: {result['correct']}")
            st.write("---")
                               
        accuracy = sum([1 for r in eval_results if r['correct']]) / len(eval_results)
        st.write(f"Model Accuracy:  {accuracy:.2%}")
                               
    st.subheader("Execution Time Monitoring")                          
    fig, ax = plt.subplots()
    timestamps =  [datetime.fromisoformat(log['timestamp']) for log in model_monitor.logs]                      
    execution_times = [log['execution_time'] for log in model_monitor.logs] 
    ax.plot(timestamps, execution_times, marker='o')
    ax.set_title("Model Execution Times")
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Execution Time (seconds)')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    avg_execution_time = model_monitor.get_average_execution_time()
    st.write(f"Average Execution Time: {average_excution_time:.2f} seconds")
    pass
    
if __name__ == '__main__':
    pass        

