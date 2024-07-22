from IPython.display import display, Markdown, Image
from langgraph.graph.graph import CompiledGraph


def print_markdown(markdown: str):
    display(Markdown(markdown))


def print_graph(graph: CompiledGraph):
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass
