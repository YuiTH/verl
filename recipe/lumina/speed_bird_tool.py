import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf

from recipe.lumina.SpeedBirdClient import SpeedbirdMcpClient
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class SpeedBirdTool(BaseTool):
    """Simple wrapper that runs SpeedBird MCP search."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        # OmegaConf may be passed in; normalize to a plain dict for convenience.
        normalized_config = OmegaConf.to_container(config, resolve=True) if not isinstance(config, dict) else config
        super().__init__(normalized_config, tool_schema)

        self.base_url = normalized_config.get("base_url", "https://www.speedbird-mcp.com").rstrip("/")
        self.mcp_path = normalized_config.get("mcp_path", "/speedbird/mcp")
        self.tenant_id = normalized_config.get("tenant_id", "72f988bf-86f1-41af-91ab-2d7cd011db47")
        self.client_id = normalized_config.get("client_id", "80d5bbf2-c435-4dff-a24d-11002e95a275")
        self.scope = normalized_config.get("scope", "api://ab2d8c9a-e6c1-4e32-a804-9bb54758ed8b/.default")
        self.keyvault_url = normalized_config.get("keyvault_url", "https://luminaftcerttest.vault.azure.net")
        self.certificate_name = normalized_config.get("certificate_name", "EvalTestCert")
        self.use_auth = bool(normalized_config.get("use_auth", True))
        self.protocol_version = normalized_config.get("protocol_version", "2025-06-18")

        session = normalized_config.get("session")

        self.tool_client = SpeedbirdMcpClient(
            base_url=self.base_url,
            mcp_path=self.mcp_path,
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            scope=self.scope,
            keyvault_url=self.keyvault_url,
            certificate_name=self.certificate_name,
            use_auth=self.use_auth,
            protocol_version=self.protocol_version,
            session=session,
        )
        logger.info("Initialized SpeedBirdTool with base_url=%s mcp_path=%s", self.base_url, self.mcp_path)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web and return a structured list of results. Produces a new search results page ref_id (e.g., turn0search0) with numbered links (0-based). Use to begin or broaden discovery.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "requests": {
                                "type": "array",
                                "description": "One or more search requests. Each request should target a distinct aspect.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "q": {
                                            "type": "string",
                                            "description": "Concise query capturing essential info needed."
                                        },
                                        "recency": {
                                            "type": "integer",
                                            "description": "Optional filter: only show results from the last X days."
                                        },
                                        "domains": {
                                            "type": "array",
                                            "description": "Optional whitelist of domains to search.",
                                            "items": {"type": "string"}
                                        },
                                        "topN": {
                                            "type": "integer",
                                            "description": "Number of results to return (default 10, max 100).",
                                            "minimum": 1,
                                            "maximum": 100
                                        },
                                        "source": {
                                            "type": "string",
                                            "description": "Backend to use (default: web_with_bing)."
                                        }
                                    },
                                    "required": ["q"]
                                }
                            }
                        },
                        "required": ["requests"]
                    }
                }
            }
        ]
        search_schema = tools[0]

        return OpenAIFunctionToolSchema(**search_schema)
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query = (
            parameters.get("query")
            or parameters.get("q")
            or parameters.get("search")
        )
        if not query:
            err = "query is required for SpeedBird search"
            logger.error(err)
            return ToolResponse(text=json.dumps({"error": err})), 0.0, {"status": "error", "error": err}

        max_results = parameters.get("max_results") or parameters.get("maxResults") or parameters.get("topN") or 10
        language = parameters.get("language", "en")
        region = parameters.get("region", "us")
        raw_domains = parameters.get("domains")
        if raw_domains is not None and not isinstance(raw_domains, (list, tuple)):
            err = "domains must be a list of strings"
            logger.error(err)
            return ToolResponse(text=json.dumps({"error": err})), 0.0, {"status": "error", "error": err}
        normalized_domains = [str(domain) for domain in raw_domains or [] if domain] or None

        try:
            result = await self.tool_client.search(
                query=str(query),
                max_results=int(max_results),
                language=language,
                region=region,
                domains=normalized_domains,
            )
            logger.info("SpeedBirdTool search success query_preview=%s", str(query)[:50])
            return ToolResponse(text=json.dumps(result)), 0.0, {"status": "success"}
        except Exception as exc:
            logger.error("SpeedBirdTool search failed for query=%s error=%s", str(query)[:50], exc)
            return ToolResponse(text=json.dumps({"error": str(exc)})), 0.0, {"status": "error", "error": str(exc)}


if __name__ == "__main__":
    import asyncio

    class _InlineResponse:
        def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
            self._payload = payload
            self.status = status

        def raise_for_status(self) -> None:
            if not (200 <= self.status < 300):
                msg = f"HTTP {self.status}"
                raise RuntimeError(msg)

        async def json(self) -> dict[str, Any]:
            return self._payload

    class _InlineRequestContext:
        def __init__(self, handler, url: str, payload: dict[str, Any], headers: dict[str, Any]) -> None:
            self._handler = handler
            self._url = url
            self._payload = payload
            self._headers = headers
            self._response: _InlineResponse | None = None

        async def __aenter__(self) -> _InlineResponse:
            response_payload = await self._handler(self._url, self._payload, self._headers)
            self._response = _InlineResponse(response_payload)
            return self._response

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            self._response = None
            return False

    class _InlineSession:
        def __init__(self, handler):
            self._handler = handler
            self._closed = False

        @property
        def closed(self) -> bool:
            return self._closed

        async def close(self) -> None:
            self._closed = True

        def post(self, url: str, json: dict[str, Any], headers: dict[str, Any]) -> _InlineRequestContext:
            return _InlineRequestContext(self._handler, url, json, headers)

    class _FakeSpeedbirdAPI:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        async def handle(self, url: str, payload: dict[str, Any], headers: dict[str, Any]) -> dict[str, Any]:
            self.requests.append({"url": url, "payload": payload, "headers": headers})
            return {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {"items": [{"title": "Result"}]},
            }

    async def _with_fake_session(test_coro):
        api = _FakeSpeedbirdAPI()
        session = _InlineSession(api.handle)
        base_url = "http://speedbird.local"
        try:
            await test_coro(api, base_url, session)
        finally:
            await session.close()

    def _create_tool(config: dict | None = None) -> SpeedBirdTool:
        return SpeedBirdTool(config or {}, None)

    def _test_schema() -> None:
        tool = _create_tool()
        schema = tool.get_openai_tool_schema()
        assert schema.type == "function", "schema type should be 'function'"
        assert schema.function.name == "search", "tool should expose the search function"
        schema_dict = schema.model_dump()
        requests_schema = schema_dict["function"]["parameters"]["properties"].get("requests") or {}
        assert requests_schema.get("type") == "array", "requests should remain an array parameter"

    async def _test_execute_real_call() -> None:
        async def _run(api: _FakeSpeedbirdAPI, base_url: str, session: _InlineSession) -> None:
            tool = _create_tool({"base_url": base_url, "use_auth": False, "session": session})
            response, reward, metrics = await tool.execute(
                instance_id="instance-1",
                parameters={
                    "q": "mars missions",
                    "topN": 5,
                    "domains": ["nasa.gov", "jpl.nasa.gov"],
                    "language": "fr",
                    "region": "ca",
                },
            )
            assert reward == 0.0
            assert metrics == {"status": "success"}
            payload = api.requests[-1]["payload"]
            arguments = payload["params"]["arguments"]
            assert arguments["query"] == "mars missions"
            assert arguments["maxResults"] == 5
            assert arguments["language"] == "fr"
            assert arguments["region"] == "ca"
            assert arguments["domains"] == ["nasa.gov", "jpl.nasa.gov"]
            assert json.loads(response.text)["result"]["items"][0]["title"] == "Result"

        await _with_fake_session(_run)

    async def _test_execute_requires_query() -> None:
        tool = _create_tool()
        response, reward, metrics = await tool.execute(instance_id="instance-2", parameters={})
        assert json.loads(response.text) == {"error": "query is required for SpeedBird search"}
        assert reward == 0.0
        assert metrics["status"] == "error"

    async def _test_execute_requires_list_domains() -> None:
        tool = _create_tool()
        response, reward, metrics = await tool.execute(
            instance_id="instance-3", parameters={"query": "mars", "domains": "nasa.gov"}
        )
        assert json.loads(response.text) == {"error": "domains must be a list of strings"}
        assert reward == 0.0
        assert metrics["status"] == "error"

    async def _test_integration_e2e() -> None:
        """End-to-end integration test that calls the real SpeedBird API."""
        config = {
            "base_url": "https://www.speedbird-mcp.com",
            "mcp_path": "/speedbird/mcp",
            "tenant_id": "72f988bf-86f1-41af-91ab-2d7cd011db47",
            "client_id": "80d5bbf2-c435-4dff-a24d-11002e95a275",
            "scope": "api://ab2d8c9a-e6c1-4e32-a804-9bb54758ed8b/.default",
            "keyvault_url": "https://luminaftcerttest.vault.azure.net",
            "certificate_name": "EvalTestCert",
            "use_auth": True,
            "protocol_version": "2025-06-18",
        }
        tool = SpeedBirdTool(config, None)

        print("Running E2E integration test against real SpeedBird API...")

        response, reward, metrics = await tool.execute(
            instance_id="e2e-test-1",
            parameters={
                "q": "latest news on climate change",
                "topN": 5,
            },
        )

        result = json.loads(response.text)
        print(f"Search response: {json.dumps(result, indent=2)}")

        if "error" in result:
            print(f"E2E test failed with error: {result['error']}")
            return

        assert metrics["status"] == "success", f"Expected success, got {metrics}"
        assert reward == 0.0, f"Expected 0.0 reward, got {reward}"

        if "result" in result:
            items = result.get("result", {}).get("items", [])
            print(f"Received {len(items)} search results")
            for i, item in enumerate(items[:3]):
                print(f"  [{i}] {item.get('title', 'No title')[:60]}")

        print("E2E integration test passed!")

    def _run_inline_tests() -> None:
        _test_schema()
        asyncio.run(_test_execute_real_call())
        asyncio.run(_test_execute_requires_query())
        asyncio.run(_test_execute_requires_list_domains())
        print("SpeedBirdTool inline tests passed.")

    def _run_integration_tests() -> None:
        asyncio.run(_test_integration_e2e())

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", action="store_true", help="Run E2E integration test against real API")
    args = parser.parse_args()

    _run_integration_tests()
    _run_inline_tests()


# Example OmegaConf YAML snippet for enabling this tool:
# tools:
#   - class_name: recipe.lumina.speed_bird_tool.SpeedBirdTool
#     tool_schema:
#       type: function
#       function:
#         name: speedbird_search
#         description: Run a SpeedBird MCP web search
#         parameters:
#           type: object
#           properties:
#             query:
#               type: string
#               description: Query text to search
#             max_results:
#               type: integer
#               description: Maximum number of results (1-20)
#             language:
#               type: string
#             region:
#               type: string
#           required: [query]
#     config:
#       base_url: "https://your-hostname"
#       mcp_path: "/speedbird/mcp"
#       tenant_id: "<tenant>"
#       client_id: "<client>"
#       scope: "api://.../.default"
#       keyvault_url: "https://your-kv.vault.azure.net"
#       certificate_name: "cert-name"
#       use_auth: true
#       protocol_version: "2025-06-18"
