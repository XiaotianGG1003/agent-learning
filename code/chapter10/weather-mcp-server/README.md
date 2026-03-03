# Weather MCP Server

真实天气查询 MCP 服务器，基于 HelloAgents 框架开发。

## 功能特性

- 🌤️ 实时天气查询
- 🌍 支持12个中国主要城市
- 🔄 使用 wttr.in API（无需密钥）
- 🚀 基于 HelloAgents 框架

## 安装

```bash
pip install hello-agents requests
```

## 使用方法

### 直接运行

```bash
python server.py
```

### 在 Claude Desktop 中使用

编辑 `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) 或 `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

### 在 HelloAgents 中使用

```python
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import MCPTool

agent = SimpleAgent(name="天气助手", llm=HelloAgentsLLM())
weather_tool = MCPTool(server_command=["python", "server.py"])
agent.add_tool(weather_tool)

response = agent.run("北京今天天气怎么样？")
```

## API 工具

### get_weather

获取指定城市的当前天气。

**参数：**
- `city` (string): 城市名称（支持中文和英文）

**示例：**
```json
{
  "city": "北京"
}
```

**返回：**
```json
{
  "city": "北京",
  "temperature": 10.0,
  "feels_like": 9.0,
  "humidity": 94,
  "condition": "Light rain",
  "wind_speed": 1.7,
  "visibility": 10.0,
  "timestamp": "2025-10-09 13:25:03"
}
```

### list_supported_cities

列出所有支持的中文城市。

**返回：**
```json
{
  "cities": ["北京", "上海", "广州", "深圳", "杭州", "成都", "重庆", "武汉", "西安", "南京", "天津", "苏州"],
  "count": 12
}
```

### get_server_info

获取服务器信息。

**返回：**
```json
{
  "name": "Weather MCP Server",
  "version": "1.0.0",
  "tools": ["get_weather", "list_supported_cities", "get_server_info"]
}
```

## 支持的城市

北京、上海、广州、深圳、杭州、成都、重庆、武汉、西安、南京、天津、苏州

也支持使用英文城市名查询全球任意城市。

## 许可证

MIT License

## 作者

HelloAgents Team

