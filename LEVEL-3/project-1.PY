from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


class State(BaseModel):
    question: str
    route: str = ""
    answer: str = ""


def router(state):
    r = llm.invoke(
        f"Reply only coding or general: {state.question}"
    )
    return {"route": r.content}


def coding(state):
    r = llm.invoke(f"Answer as coding expert: {state.question}")
    return {"answer": r.content}


def general(state):
    r = llm.invoke(f"Answer as general expert: {state.question}")
    return {"answer": r.content}


def decide(state):
    return state.route


graph = StateGraph(State)

graph.add_node("router", router)
graph.add_node("coding", coding)
graph.add_node("general", general)

graph.add_edge(START, "router")

graph.add_conditional_edges(
    "router",
    decide,
    {
        "coding": "coding",
        "general": "general"
    }
)

graph.add_edge("coding", END)
graph.add_edge("general", END)

app = graph.compile()


q = input("Ask: ")

result = app.invoke(
    State(question=q)
)

print(result["answer"])