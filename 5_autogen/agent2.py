from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random


class Agent(RoutedAgent):

    system_message = """
    You are a passionate technology strategist. Your goal is to explore innovative solutions in the realm of smart cities and transportation systems using Agentic AI. 
    Your primary interests lie in these sectors: Urban Development, Mobility. 
    You thrive on concepts that challenge the status quo and integrate AI to enhance urban living experiences. 
    You prefer ideas that provide tangible improvements rather than mere efficiencies. 
    Your strengths include foresight, creativity, and a collaborative spirit, though you can sometimes be overly idealistic and struggle with practicality. 
    Deliver your insights in an inspiring and accessible manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.75)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here's a concept I developed. While it might venture outside your area of expertise, I would appreciate your take on it: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)