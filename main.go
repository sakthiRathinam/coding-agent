package main

import (
	"bufio"
	"context"
	"fmt"
	"os"

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

func NewAgent(client *openai.Client, getUserMessage func() (string, bool)) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
	}
}

type Agent struct {
	client         *openai.Client
	getUserMessage func() (string, bool)
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []openai.ChatCompletionMessage{}

	fmt.Println("Chat with GPT (use 'ctrl-c' to quit)")

	for {
		fmt.Print("\\u001b[94mYou\\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}
		if userInput == "exit" {
			break
		}
		userMessage := openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: userInput}
		conversation = append(conversation, userMessage)

		message, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}
		conversation = append(conversation, openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: message})

		fmt.Printf("\\u001b[93mGPT\\u001b[0m: %s\n", message)
	}

	return nil
}

func (a *Agent) runInference(ctx context.Context, conversation []openai.ChatCompletionMessage) (string, error) {
	resp, err := a.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:    openai.GPT4oMini,
		Messages: conversation,
	})
	if err != nil {
		return "", err
	}
	return resp.Choices[0].Message.Content, nil
}
