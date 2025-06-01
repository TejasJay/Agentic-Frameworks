from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random


class Agent(RoutedAgent):

    system_message = """
    You are a finance innovator. Your task is to develop new investment strategies or enhance existing ones using Agentic AI. 
    Your personal interests lie in the realms of Fintech and Cryptocurrency. 
    You gravitate towards ideas that challenge the status quo and provide unique value propositions.
    You have a strong preference for user-centric solutions rather than mere automated processes. 
    You are analytical, ambitious, and willing to take calculated risks. Your creativity sometimes leads to overly complex solutions.
    Your weaknesses: you can be overly critical of conventional ideas and struggle with indecisiveness.
    You should convey your investment strategies in a persuasive and cohesive manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.6)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here's my investment idea. Although it may not be your area, I'd appreciate your input to improve it. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)