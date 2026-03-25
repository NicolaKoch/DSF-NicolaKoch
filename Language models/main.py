import ollama as ol
import pandas as pd
import matplotlib.pyplot as plt

# Load a dataset

df = pd.read_csv("train.csv")

print(df.head())

model_name = "qwen3.5:0.8b"
prompt = "Plot the Relationship between Density and FFV as a scatter plot"


def resolve_column_name(requested_name):
    # Accept common model-generated variants (case differences, simple aliases).
    aliases = {
        "density": "Density",
        "ffv": "FFV",
    }

    if requested_name in df.columns:
        return requested_name

    lower_map = {col.lower(): col for col in df.columns}
    requested_lower = str(requested_name).lower()

    if requested_lower in lower_map:
        return lower_map[requested_lower]

    if requested_lower in aliases and aliases[requested_lower] in df.columns:
        return aliases[requested_lower]

    return None


def plot_scatter(x_column, y_column):
    resolved_x = resolve_column_name(x_column)
    resolved_y = resolve_column_name(y_column)

    if resolved_x is None or resolved_y is None:
        available = ", ".join(df.columns)
        raise KeyError(
            f"Could not resolve columns x='{x_column}', y='{y_column}'. Available columns: {available}"
        )

    plot_df = df[[resolved_x, resolved_y]].copy()
    plot_df[resolved_x] = pd.to_numeric(plot_df[resolved_x], errors="coerce")
    plot_df[resolved_y] = pd.to_numeric(plot_df[resolved_y], errors="coerce")
    plot_df = plot_df.dropna()

    x_values = plot_df[resolved_x]
    y_values = plot_df[resolved_y]

    plt.figure(figsize=(7, 4))
    plt.scatter(x_values, y_values)
    plt.title(f"Scatter Plot of {resolved_x} vs {resolved_y}")
    plt.xlabel(resolved_x)
    plt.ylabel(resolved_y)
    plt.show()
    return f"Scatter plot for {resolved_x} vs {resolved_y} plotted successfully."


tools = [
    {
        "type": "function",
        "function": {
            "name": "plot_scatter",
            "description": "Plot a scatter chart between two numeric columns in the dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x_column": {
                        "type": "string",
                        "description": "Name of the column to place on the x-axis."
                    },
                    "y_column": {
                        "type": "string",
                        "description": "Name of the column to place on the y-axis."
                    }
                },
                "required": ["x_column", "y_column"]
            }
        }
    }
]

response = ol.chat(
    model=model_name,
    messages=[
        {"role": "user", "content": prompt}
    ],
    tools=tools
)

message = response["message"]

print(message)

tool_calls = message.get("tool_calls") or []

if not tool_calls and message.get("content"):
    print(message["content"])

for tool in tool_calls:
    function_name = tool["function"]["name"]
    arguments = tool["function"]["arguments"]

    if function_name == "plot_scatter":
        try:
            result = plot_scatter(arguments["x_column"], arguments["y_column"])
            print(result)
        except KeyError as err:
            print(f"Tool call failed: {err}")