**pre-requisites** 


```python
import anthropic
import os
api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic()

# open meeting transcripts
with open("docs/meeting_transcript.txt", 'r') as file:
    # Read the contents of the file
    meeting_transcript = file.read()

# set attendees
attendees = ["Sarah", "Michael", "Priya", "John"]
```

**step 1 - define your JSON template**


```python
JSON_TEMPLATE = """
{
  "action_items": [
    {
      "person": "Name or Group",
      "tasks": [
        {
          "action": "Specific task description",
          "completion_target": "Deadline or target date in dd/mm/yyy format",
          "source": {
            "person": "Name of the person who assigned the task",
            "quote": "Exact text string from meeting or document verifying the task"
          }
        },
        {
          "action": "Another task description",
          "completion_target": "Another deadline or target date",
          "source": {
            "person": "Name of the person who assigned the task",
            "quote": "Exact text string from meeting or document verifying the task"
          }
        }
        // Additional tasks can be added here with their respective sources
      ]
    }
    // Additional persons or groups with their tasks can be added here
  ]
}
"""
```

**step 2/3 - write this as a pydantic clas with validation decorators**


```python
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator
from typing import List
from datetime import datetime

class Source(BaseModel):
    person: str
    quote: str
    
    @field_validator('quote')
    @classmethod
    def quote_in_transcript(cls, v: str, info: ValidationInfo) -> str:
        if 'transcript' not in info.context:
            raise ValueError("Transcript not provided in the context.")
        if v not in info.context['transcript']:
            raise ValueError(f"Quote '{v}' not found in the meeting transcript.")
        return v

class Task(BaseModel):
    action: str
    completion_target: str
    source: Source
    
    @field_validator('completion_target')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, '%d/%m/%Y')
        except ValueError:
            raise ValueError(f"Invalid date format. Expected format: 'dd/mm/yyyy', got '{v}'")
        return v

class ActionItem(BaseModel):
    person: str
    tasks: List[Task]
    
    @field_validator('person')
    @classmethod
    def person_in_attendees(cls, v: str, info: ValidationInfo) -> str:
        if 'attendees' not in info.context:
            raise ValueError("Attendees not provided in the context.")
        if v not in info.context['attendees']:
            raise ValueError(f"Person '{v}' not found in the meeting attendees.")
        return v

class ActionItemList(BaseModel):
    action_items: List[ActionItem]
    
    @field_validator('action_items')
    @classmethod
    def validate_action_items(cls, v: List[ActionItem], info: ValidationInfo) -> List[ActionItem]:
        if 'attendees' not in info.context:
            raise ValueError("Attendees not provided in the context.")
        if 'transcript' not in info.context:
            raise ValueError("Transcript not provided in the context.")
        return v
```

**step 4 - extract JSON using LLM**


```python
# prompt template to extract action items in JSON format
EXTRACT_MEETING_INFO = f"""
<role>
you are a meticulous assistant who extracts action items from meetings in
structured JSON format
</role>

<json_template>
use the following json template:
{JSON_TEMPLATE}
</json_template>
"""
```


```python
model_id = "claude-3-haiku-20240307"

message = client.messages.create(
    model=model_id,
    max_tokens=2048,
    temperature=0,
    system=f"{EXTRACT_MEETING_INFO}",
    messages=[
        {"role": "user", "content": f"""<meeting_transcript>
                                          {meeting_transcript}
                                        </meeting_transcript>?"""},
        {"role": "assistant", "content": "{"}, # notice this part
    ],
)

json_output = "{" + message.content[0].text
```


```python
print(json_output)
```

    {
      "action_items": [
        {
          "person": "Sarah",
          "tasks": [
            {
              "action": "Update the marketing plan to focus on Instagram reels over LinkedIn articles, and reallocate the budget accordingly",
              "completion_target": "05/07/2024",
              "source": {
                "person": "John",
                "quote": "Sarah, please update the marketing plan and budget accordingly by end of week and send it over to me for review."
              }
            }
          ]
        },
        {
          "person": "Michael",
          "tasks": [
            {
              "action": "Get approval from legal on the revised website copy",
              "completion_target": "30/06/2024",
              "source": {
                "person": "John",
                "quote": "The only outstanding item is getting approval on the revised copy from legal. I sent it over to them last week and am waiting to hear back."
              }
            },
            {
              "action": "Schedule a meeting with the web development team to walk through the final website designs",
              "completion_target": "08/05/2024",
              "source": {
                "person": "John",
                "quote": "Also, can you set up a meeting with the web development team to walk through the final designs and confirm everything is in order?"
              }
            }
          ]
        },
        {
          "person": "Priya",
          "tasks": [
            {
              "action": "Put together a brief overview of the event sponsorship package and circulate it to the team",
              "completion_target": "02/05/2024",
              "source": {
                "person": "John",
                "quote": "Can you put together a brief overview of what's included in the sponsorship package and circulate it to the team? I want to make sure we're fully capitalizing on all the benefits."
              }
            },
            {
              "action": "Start working on the logistics for the event booth setup and staffing, and coordinate with Sarah on the marketing collateral",
              "completion_target": "30/06/2024",
              "source": {
                "person": "Priya",
                "quote": "I'm also going to start working on the logistics for the booth setup and staffing. I'll coordinate with Sarah on getting the marketing collateral together."
              }
            }
          ]
        }
      ]
    }


*replace one of the quotes with false quote from transcript*


```python
import re
replacement = "Sarah, update the marketing plan by today" # inject false quote into JSON for testing

# regex to find and replace the first occurrence of the "quote" value
modified_json_output = re.sub(r'"quote": ".*?"', f'"quote": "{replacement}"',
                               json_output, 
                               count=1)

print(modified_json_output)
```

    {
      "action_items": [
        {
          "person": "Sarah",
          "tasks": [
            {
              "action": "Update the marketing plan to focus on Instagram reels over LinkedIn articles, and reallocate the budget accordingly",
              "completion_target": "05/07/2024",
              "source": {
                "person": "John",
                "quote": "Sarah, update the marketing plan by today"
              }
            }
          ]
        },
        {
          "person": "Michael",
          "tasks": [
            {
              "action": "Get approval from legal on the revised website copy",
              "completion_target": "30/06/2024",
              "source": {
                "person": "John",
                "quote": "The only outstanding item is getting approval on the revised copy from legal. I sent it over to them last week and am waiting to hear back."
              }
            },
            {
              "action": "Schedule a meeting with the web development team to walk through the final website designs",
              "completion_target": "08/05/2024",
              "source": {
                "person": "John",
                "quote": "Also, can you set up a meeting with the web development team to walk through the final designs and confirm everything is in order?"
              }
            }
          ]
        },
        {
          "person": "Priya",
          "tasks": [
            {
              "action": "Put together a brief overview of the event sponsorship package and circulate it to the team",
              "completion_target": "02/05/2024",
              "source": {
                "person": "John",
                "quote": "Can you put together a brief overview of what's included in the sponsorship package and circulate it to the team? I want to make sure we're fully capitalizing on all the benefits."
              }
            },
            {
              "action": "Start working on the logistics for the event booth setup and staffing, and coordinate with Sarah on the marketing collateral",
              "completion_target": "30/06/2024",
              "source": {
                "person": "Priya",
                "quote": "I'm also going to start working on the logistics for the booth setup and staffing. I'll coordinate with Sarah on getting the marketing collateral together."
              }
            }
          ]
        }
      ]
    }


**step 5 - fix JSON with pydantic feedback**


```python
import json

max_retries = 3
retry_count = 0

# Make sure meeting_transcript is defined and contains the actual transcript string
# Specify the path to the text file in the "docs" folder
file_path = "docs/meeting_transcript.txt"

# Open the file in read mode
with open(file_path, "r") as file:
    # Read the contents of the file
    meeting_transcript = file.read()

while retry_count < max_retries:
    try:
        # Validate the JSON data using ActionItemList
        action_item_list = ActionItemList.model_validate_json(modified_json_output, context={'attendees': attendees, "transcript": meeting_transcript})
        print("Action item list parsed successfully!")
        print(action_item_list)
        break  # Exit the loop if validation succeeds

    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Error: {str(e)}")

        # Extract the error message
        if isinstance(e, json.JSONDecodeError):
            error_message = f"Invalid JSON: {str(e)}"
        else:
            error_message = f"Validation error: {str(e)}"

        # Increment the retry count
        retry_count += 1

        # Check if the maximum number of retries is reached
        if retry_count == max_retries:
            print("Maximum number of retries reached. Unable to fix the JSON.")
            break

        # Prepare the input for the Claude model
        messages = [
            {"role": "user", "content": f"""<meeting_transcript>
                                                {meeting_transcript}
                                            </meeting_transcript>
                                            <invalid_json>
                                                {modified_json_output}
                                            </invalid_json>
                                            <errors>
                                                {error_message}
                                            </errors>
                                            """},
            {"role": "assistant", "content": "{"},
        ]

        # Call the Claude model to fix the JSON
        message = client.messages.create(
            model=model_id,
            max_tokens=2048,
            temperature=0,
            system=f"{EXTRACT_MEETING_INFO}",
            messages=messages,
        )

        # Update the JSON output with the fixed JSON
        modified_json_output = "{" + message.content[0].text
```

    Error: 1 validation error for ActionItemList
    action_items.0.tasks.0.source.quote
      Value error, Quote 'Sarah, update the marketing plan by today' not found in the meeting transcript. [type=value_error, input_value='Sarah, update the marketing plan by today', input_type=str]
        For further information visit https://errors.pydantic.dev/2.7/v/value_error
    Action item list parsed successfully!
    action_items=[ActionItem(person='Sarah', tasks=[Task(action='Update the marketing plan to focus on Instagram reels over LinkedIn articles, and reallocate the budget accordingly', completion_target='05/07/2024', source=Source(person='John', quote='Sarah, please update the marketing plan and budget accordingly by end of week and send it over to me for review.'))]), ActionItem(person='Michael', tasks=[Task(action='Get approval from legal on the revised website copy', completion_target='30/06/2024', source=Source(person='John', quote='The only outstanding item is getting approval on the revised copy from legal. I sent it over to them last week and am waiting to hear back.')), Task(action='Schedule a meeting with the web development team to walk through the final website designs', completion_target='08/05/2024', source=Source(person='John', quote='Also, can you set up a meeting with the web development team to walk through the final designs and confirm everything is in order?'))]), ActionItem(person='Priya', tasks=[Task(action='Put together a brief overview of the event sponsorship package and circulate it to the team', completion_target='02/05/2024', source=Source(person='John', quote="Can you put together a brief overview of what's included in the sponsorship package and circulate it to the team? I want to make sure we're fully capitalizing on all the benefits.")), Task(action='Start working on the logistics for the event booth setup and staffing, and coordinate with Sarah on the marketing collateral', completion_target='30/06/2024', source=Source(person='Priya', quote="I'm also going to start working on the logistics for the booth setup and staffing. I'll coordinate with Sarah on getting the marketing collateral together."))])]



```python
# voila
print(modified_json_output)
```

    {
      "action_items": [
        {
          "person": "Sarah",
          "tasks": [
            {
              "action": "Update the marketing plan to focus on Instagram reels over LinkedIn articles, and reallocate the budget accordingly",
              "completion_target": "05/07/2024",
              "source": {
                "person": "John",
                "quote": "Sarah, please update the marketing plan and budget accordingly by end of week and send it over to me for review."
              }
            }
          ]
        },
        {
          "person": "Michael",
          "tasks": [
            {
              "action": "Get approval from legal on the revised website copy",
              "completion_target": "30/06/2024",
              "source": {
                "person": "John",
                "quote": "The only outstanding item is getting approval on the revised copy from legal. I sent it over to them last week and am waiting to hear back."
              }
            },
            {
              "action": "Schedule a meeting with the web development team to walk through the final website designs",
              "completion_target": "08/05/2024",
              "source": {
                "person": "John",
                "quote": "Also, can you set up a meeting with the web development team to walk through the final designs and confirm everything is in order?"
              }
            }
          ]
        },
        {
          "person": "Priya",
          "tasks": [
            {
              "action": "Put together a brief overview of the event sponsorship package and circulate it to the team",
              "completion_target": "02/05/2024",
              "source": {
                "person": "John",
                "quote": "Can you put together a brief overview of what's included in the sponsorship package and circulate it to the team? I want to make sure we're fully capitalizing on all the benefits."
              }
            },
            {
              "action": "Start working on the logistics for the event booth setup and staffing, and coordinate with Sarah on the marketing collateral",
              "completion_target": "30/06/2024",
              "source": {
                "person": "Priya",
                "quote": "I'm also going to start working on the logistics for the booth setup and staffing. I'll coordinate with Sarah on getting the marketing collateral together."
              }
            }
          ]
        }
      ]
    }



```python

```
