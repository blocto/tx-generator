from IPython.display import display, Markdown, Image, update_display
from langgraph.graph.graph import CompiledGraph

from langchain_core.messages import HumanMessage


class NotebookPrinter:

    def __init__(self, generator):
        self.generator = generator

    def _transform_message(self, user_input: str):
        return {"messages": [HumanMessage(content=user_input)]}

    async def ask(self, question: str):
        transformed = self._transform_message(question)
        response = await self.generator.ainvoke(transformed)
        return display(Markdown(response["messages"][-1].content))

    async def ask_streaming(self, question: str):
        display_id = "streaming_output"
        display(Markdown(""), display_id=display_id)
        accumulated_content = ""
        transformed = self._transform_message(question)
        async for event in self.generator.astream_events(transformed, version="v1"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    accumulated_content += content
                    update_display(Markdown(accumulated_content), display_id=display_id)
            elif kind == "on_tool_start":
                print(
                    f"Using tool: {event['name']} with inputs: {event['data'].get('input')}"
                )
                print("--")
            elif kind == "on_tool_end":
                print(f"Done tool: {event['name']}")
                # print(f"Tool output was: {event['data'].get('output')}")
                print("--")
            elif kind == "on_retriever_start":
                print("Using retriever")
                # print(
                #     f"Using retriever: {event['name']} with inputs: {event['data'].get('input')}"
                # )
                print("--")
            elif kind == "on_retriever_end":
                output = event["data"].get("output")
                docs = output["documents"]
                print(f"Retrieved documents: {[doc.metadata for doc in docs]}")
                print("--")

    def print_graph(self, graph: CompiledGraph):
        try:
            display(Image(graph.get_graph().draw_mermaid_png()))
        except Exception:
            # This requires some extra dependencies and is optional
            pass
