"""Sanity: FastMCP tool registration (in-memory client) + job cancel."""

import asyncio
import sys
import time

sys.path.insert(0, "src")

from fastmcp import Client  # noqa: E402

from iramuteq_mcp import mcp  # noqa: E402


async def main():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("tools:", sorted(t.name for t in tools))
        res = await client.call_tool("status", {})
        print("status ready:", res.data.get("ready"), "| R:", res.data.get("r_path"))

        # submit a job then cancel it immediately
        res = await client.call_tool(
            "corpus_alceste",
            {"path": "tests/fixtures/mini_corpus.txt", "nbcl_p1": 3, "max_actives": 200},
        )
        job_id = res.data["job_id"]
        await client.call_tool("job_cancel", {"job_id": job_id})
        for _ in range(60):
            st = (await client.call_tool("job_status", {"job_id": job_id})).data
            if st["status"] not in ("queued", "running"):
                break
            time.sleep(0.5)
        print("cancelled job status:", st["status"], "| error:", (st.get("error") or "")[:120])

        hist = (await client.call_tool("analysis_list", {})).data
        print("history corpora:", len(hist.get("corpora", [])))


if __name__ == "__main__":
    asyncio.run(main())
