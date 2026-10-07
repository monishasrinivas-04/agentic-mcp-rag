import asyncio

from mcp import Client

from mcp_servers.repository_server import mcp


async def main():

    async with Client(mcp) as client:

        print("\nMCP connection established.")

        # --------------------------------
        # List available tools
        # --------------------------------

        tools = await client.list_tools()

        print("\nAvailable tools:")

        for tool in tools.tools:
            print(f"- {tool.name}")

        # --------------------------------
        # Test list_files
        # --------------------------------

        print("\nTesting list_files()...")

        result = await client.call_tool(
            "list_files",
            {}
        )

        print("\nRepository files:")

        print(result)

        # --------------------------------
        # Test read_file
        # --------------------------------

        print("\nTesting read_file()...")

        result = await client.call_tool(
            "read_file",
            {
                "file_path": "README.md"
            }
        )

        print("\nREADME:")

        print(result)

        # --------------------------------
        # Test search_repository
        # --------------------------------

        print(
            "\nTesting search_repository()..."
        )

        result = await client.call_tool(
            "search_repository",
            {
                "query": "rooftop temperature",
                "max_results": 3
            }
        )

        print("\nSearch results:")

        print(result)


if __name__ == "__main__":

    asyncio.run(main())