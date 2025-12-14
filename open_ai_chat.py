from __future__ import annotations
import asyncio
import os
import json
import sys
from typing import Any

from dotenv import load_dotenv
from fastmcp import Client
from openai import OpenAI


def _build_openai_tools_schema(tools: list[Any])-> list[dict[str,Any]]:
    schemas: list[dict[str,Any]] = []
    for t in tools:
        name=getattr(t,"name","None") or "tool"
        description=getattr(t,"description","None") or getattr(t,"title","None") or ""
        params = getattr(t,"inputSchema",None) or {"type":"object","properties":{}}
        schemas.append(
            {
                "type": "function",
                "function":{
                    "name":name,
                    "description":description,
                    "parameters":params,
                }
            }
        )
    return schemas

def _tool_result_to_text(result:Any)->str:
    """Convert fastmcp ToolCallResult to a plain text for OpenAI tool message."""
    try:
        sc=getattr(result,"structured_content",None) or getattr(result,"structuredContent",None)
        if sc is not None:
            return json.dumps(sc)   
        
        content=getattr(result,"content",None)
        if isinstance(content,(list,tuple)):
            texts: list[str]=[]
            for block in content:
                if isinstance(block,dict):
                    if block.get("type") == "text":
                        text.append(block.get("text",""))
                else:
                    t=getattr(block,"text",None)
                    if isinstance(t,str):
                        texts.append(t)
            
            if texts:
                return "\n".join(texts)
        
        return str(result)
    except Exception as e:
        return str(result)


async def main()-> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env file.")

    server_url= os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp")

    user_text= "what is temperature in benguluru today?"

    if len(sys.argv) > 1:
        user_text = " ".join(sys.argv[1:])

    async with Client(server_url) as mcp_client:
        tools = await mcp_client.list_tools()
        openai_tools = _build_openai_tools_schema(tools)
        print(openai_tools)

        client = OpenAI(api_key=api_key)

        messages= [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_text}
        ]

        response =client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
            temperature=0.2,
        )

        choice = response.choices[0]

        msg = choice.message

        if msg.content:
            print("Assistant:", msg.content)

        if getattr(msg, "tool_calls", None):
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": msg.tool_calls
            })

            for tc in msg.tool_calls:
                if tc.type == "funtion":
                    name =tc.fuction.name
                    arguments = tc. function.arguments or "{}"
                    try:
                        args = json.loads(arguments)
                    except Exception:
                        arg={}
                    print(f"\n[Executing tool] {name}({args})")

                    result = await mcp_client.call_tool(name,args)
                    print("Tool result:", result)

                    messages.append({
                        "role": "tool",
                        "tool_call_id":tc.id,
                        "content": _tool_result_to_text(result),
                    })

            final = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.2,
            )
            print("\Final:", final.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())