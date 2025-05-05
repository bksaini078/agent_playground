from agno.agent import Agent
from agno.tools.email import EmailTools
from agno.models.azure import AzureOpenAI


receiver_email = ""
sender_email = ""
sender_name = "Bhupender Kumar Saini"
sender_passkey = "<sender_passkey>"

agent = Agent(
    model=AzureOpenAI(id="gpt-4o-2024-08-06",
                      api_version="2024-02-01",
                      azure_endpoint="",
                      api_key=""),
    tools=[
        EmailTools(
            receiver_email=receiver_email,
            sender_email=sender_email,
            sender_name=sender_name,
            sender_passkey=sender_passkey,
        )
    ],
    description="You are an email agent that helps users send emails.",
    instructions=[
        "Ensure the email is sent to the correct recipient.",
        "Include a subject line in your email.",
        "Confirm the email has been sent successfully.",
    ]
)

agent.print_response("send an email to sonalsaini29@gmail.com with subject 'Test Email' and body 'This is a test email.'",)