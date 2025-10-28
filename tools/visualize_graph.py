# visualize_graph.py
from pathlib import Path
from langchain_core.runnables.graph import MermaidDrawMethod, CurveStyle
from reddit_langgraph_agent import build_agent_graph, DeepSeekR1Orchestrator


# No-op stand-ins to avoid starting keyboard listeners/threads
class NoopKillSwitch:
    def start(self):
        pass

    def stop(self):
        pass

    def triggered(self):
        return False


class NoopContinueWaiter:
    def wait(self, *_, **__):
        return True

    def stop(self):
        pass


def main():
    orchestrator = DeepSeekR1Orchestrator()
    kill = NoopKillSwitch()
    cont = NoopContinueWaiter()

    builder = build_agent_graph(
        orchestrator=orchestrator,
        kill_switch=kill,
        continue_waiter=cont,
        use_perplexity=False,  # doesn’t matter for topology
    )
    compiled = builder.compile()  # CompiledStateGraph
    g = compiled.get_graph()  # Graph object for drawing

    # Produce a Mermaid PNG (uses mermaid.ink by default)
    png = g.draw_mermaid_png(
        draw_method=MermaidDrawMethod.API, curve_style=CurveStyle.LINEAR
    )
    out = Path("reddit_agent_graph.png")
    out.write_bytes(png)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
