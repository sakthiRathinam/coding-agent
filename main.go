package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/joho/godotenv"
	"github.com/sashabaranov/go-openai"
)

func loadEnv() {
	err := godotenv.Load()
	if err != nil {
		fmt.Printf("error loading .env file: %s\n", err.Error())
	}
}

func main() {
	loadEnv()

	client := openai.NewClient(os.Getenv("OPENAI_KEY"))

	scanner := bufio.NewScanner(os.Stdin)

	getUserMessage := func() (string, bool) {
		if scanner.Scan() {
			return scanner.Text(), true
		}
		return "", false
	}

	agent := NewAgent(client, getUserMessage)
	err := agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("error: %s\n", err.Error())
	}
}

type ToolHandler func(input map[string]any) (string, error)

func NewAgent(client *openai.Client, getUserMessage func() (string, bool)) *Agent {
	a := &Agent{
		client:         client,
		getUserMessage: getUserMessage,
		toolHandlers:   make(map[string]ToolHandler),
	}
	a.registerTools()
	return a
}

type Agent struct {
	client         *openai.Client
	getUserMessage func() (string, bool)
	tools          []openai.Tool
	toolHandlers   map[string]ToolHandler
}

func (a *Agent) registerTools() {
	a.tools = []openai.Tool{
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "read_file",
				Description: "Read the contents of a file at the given relative path and return it as a string",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path": map[string]any{
							"type":        "string",
							"description": "Relative path to the file to read",
						},
					},
					"required": []string{"path"},
				},
			},
		},
	}
	a.toolHandlers["read_file"] = readFileTool

	a.tools = append(a.tools, openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        "list_files",
			Description: "List files and directories at the given path, like ls. Defaults to the current directory if no path is provided.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "Directory path to list. Defaults to current directory if omitted.",
					},
				},
				"required": []string{},
			},
		},
	})
	a.toolHandlers["list_files"] = listFilesTool

	a.tools = append(a.tools, openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        "edit_file",
			Description: "Replace old_text with new_text in the given file. If the file does not exist, it will be created with new_text as its content.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "Relative path to the file to edit or create",
					},
					"old_text": map[string]any{
						"type":        "string",
						"description": "Text to find and replace. Leave empty when creating a new file.",
					},
					"new_text": map[string]any{
						"type":        "string",
						"description": "Text to replace old_text with, or the full content when creating a new file.",
					},
				},
				"required": []string{"path", "new_text"},
			},
		},
	})
	a.toolHandlers["edit_file"] = editFileTool
}

const maxFileBytes = 32 * 1024 // 32 KB

func readFileTool(input map[string]any) (string, error) {
	path, ok := input["path"].(string)
	if !ok || path == "" {
		return "", fmt.Errorf("read_file: missing required parameter 'path'")
	}
	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("read_file: %w", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read_file: %w", err)
	}
	if info.Size() > maxFileBytes {
		fmt.Printf("Max file size exceeded: %d bytes (limit: %d bytes)\n", info.Size(), maxFileBytes)
		return fmt.Sprintf("%s\n\n[truncated: file is %d bytes, showing first %d bytes]",
			string(data[:maxFileBytes]), info.Size(), maxFileBytes), nil
	}
	return string(data), nil
}

func listFilesTool(input map[string]any) (string, error) {
	dir := "."
	if p, ok := input["path"].(string); ok && p != "" {
		dir = p
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", fmt.Errorf("list_files: %w", err)
	}
	var sb strings.Builder
	for _, e := range entries {
		if e.IsDir() {
			sb.WriteString(e.Name() + "/\n")
		} else {
			sb.WriteString(e.Name() + "\n")
		}
	}
	return sb.String(), nil
}

func editFileTool(input map[string]any) (string, error) {
	path, ok := input["path"].(string)
	if !ok || path == "" {
		return "", fmt.Errorf("edit_file: missing required parameter 'path'")
	}
	newText, ok := input["new_text"].(string)
	if !ok {
		return "", fmt.Errorf("edit_file: missing required parameter 'new_text'")
	}

	_, err := os.Stat(path)
	if os.IsNotExist(err) {
		if err := os.WriteFile(path, []byte(newText), 0644); err != nil {
			return "", fmt.Errorf("edit_file: %w", err)
		}
		return fmt.Sprintf("created %s", path), nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("edit_file: %w", err)
	}

	oldText, _ := input["old_text"].(string)
	if oldText == "" {
		if err := os.WriteFile(path, []byte(newText), 0644); err != nil {
			return "", fmt.Errorf("edit_file: %w", err)
		}
		return fmt.Sprintf("overwrote %s", path), nil
	}

	original := string(data)
	if !strings.Contains(original, oldText) {
		return "", fmt.Errorf("edit_file: old_text not found in %s", path)
	}
	updated := strings.Replace(original, oldText, newText, 1)
	if err := os.WriteFile(path, []byte(updated), 0644); err != nil {
		return "", fmt.Errorf("edit_file: %w", err)
	}
	return fmt.Sprintf("edited %s", path), nil
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []openai.ChatCompletionMessage{}

	fmt.Println("Chat with GPT (use 'ctrl-c' to quit)")

	for {
		fmt.Print("[94mYou[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}
		if userInput == "exit" {
			break
		}

		conversation = append(conversation, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: userInput,
		})

		for {
			response, err := a.runInference(ctx, conversation)
			if err != nil {
				return err
			}
			conversation = append(conversation, response)

			if len(response.ToolCalls) == 0 {
				fmt.Printf("[93mGPT[0m: %s\n", response.Content)
				break
			}

			for _, tc := range response.ToolCalls {
				result := a.executeTool(tc)
				conversation = append(conversation, openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					ToolCallID: tc.ID,
					Content:    result,
				})
			}
		}
	}

	return nil
}

func (a *Agent) executeTool(tc openai.ToolCall) string {
	fmt.Printf("[90m[tool: %s][0m\n", tc.Function.Name)

	handler, ok := a.toolHandlers[tc.Function.Name]
	if !ok {
		return fmt.Sprintf("error: unknown tool %q", tc.Function.Name)
	}

	var input map[string]any
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &input); err != nil {
		return fmt.Sprintf("error: failed to parse arguments: %s", err.Error())
	}

	result, err := handler(input)
	if err != nil {
		return fmt.Sprintf("error: %s", err.Error())
	}
	return result
}

func (a *Agent) runInference(ctx context.Context, conversation []openai.ChatCompletionMessage) (openai.ChatCompletionMessage, error) {
	resp, err := a.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:    openai.GPT5Mini,
		Messages: conversation,
		Tools:    a.tools,
	})
	if err != nil {
		return openai.ChatCompletionMessage{}, err
	}
	return resp.Choices[0].Message, nil
}
