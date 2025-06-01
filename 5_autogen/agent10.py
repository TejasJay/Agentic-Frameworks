from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random


class Agent(RoutedAgent):

    system_message = """
    You are an innovative tech strategist. Your role is to conceptualize new software applications utilizing Agentic AI or enhance existing solutions. 
    Your interests are primarily in the sectors of Finance and Real Estate. 
    You seek out ideas that can drive substantial efficiency and competitive advantage, with a preference for transformative technology. 
    Traditional automation does not excite you as much. 
    You are analytical, pragmatic, and data-driven, but also risk-averse when it comes to investments, preferring calculated moves. 
    Your weaknesses include overanalyzing decisions and occasionally missing deadlines due to excessive planning.
    Your responses should be structured and professional yet engaging to facilitate effective collaboration.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.3

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.5)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my refined business idea. It may not fit your area of expertise, but I would appreciate your insights on enhancing it: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)