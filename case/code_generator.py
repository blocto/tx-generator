from langchain_openai import ChatOpenAI


if __name__ == "__main__":
    model = ChatOpenAI(model="gpt-4o-mini")
    result = model.invoke("hello")
    print(result.content)
