# Generated by ariadne-codegen
# Source: tools/graphql_codegen/automations/

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional, Union

from pydantic import Field
from typing_extensions import Annotated

from wandb._pydantic import GQLBase, GQLId, Typename

from .enums import AlertSeverity, EventTriggeringConditionType


class ArtifactPortfolioScopeFields(GQLBase):
    typename__: Typename[Literal["ArtifactPortfolio"]] = "ArtifactPortfolio"
    id: GQLId
    name: str


class ArtifactSequenceScopeFields(GQLBase):
    typename__: Typename[Literal["ArtifactSequence"]] = "ArtifactSequence"
    id: GQLId
    name: str


class CreateFilterTriggerResult(GQLBase):
    typename__: Typename[Literal["CreateFilterTriggerPayload"]] = (
        "CreateFilterTriggerPayload"
    )
    trigger: Optional[TriggerFields]


class DeleteTriggerResult(GQLBase):
    typename__: Typename[Literal["DeleteTriggerPayload"]] = "DeleteTriggerPayload"
    success: bool


class FilterEventFields(GQLBase):
    typename__: Typename[Literal["FilterEventTriggeringCondition"]] = (
        "FilterEventTriggeringCondition"
    )
    event_type: EventTriggeringConditionType = Field(alias="eventType")
    filter: str


class GenericWebhookActionFields(GQLBase):
    typename__: Typename[Literal["GenericWebhookTriggeredAction"]] = (
        "GenericWebhookTriggeredAction"
    )
    integration: Union[
        GenericWebhookActionFieldsIntegrationIntegration,
        GenericWebhookActionFieldsIntegrationGenericWebhookIntegration,
    ] = Field(discriminator="typename__")
    request_payload: Optional[str] = Field(alias="requestPayload")


class GenericWebhookActionFieldsIntegrationIntegration(GQLBase):
    typename__: Typename[
        Literal["GitHubOAuthIntegration", "Integration", "SlackIntegration"]
    ]


class GenericWebhookIntegrationConnectionFields(GQLBase):
    typename__: Typename[Literal["IntegrationConnection"]] = "IntegrationConnection"
    page_info: PageInfoFields = Field(alias="pageInfo")
    edges: List[GenericWebhookIntegrationConnectionFieldsEdges]


class GenericWebhookIntegrationConnectionFieldsEdges(GQLBase):
    cursor: str
    node: Optional[
        Annotated[
            Union[
                GenericWebhookIntegrationConnectionFieldsEdgesNodeIntegration,
                GenericWebhookIntegrationConnectionFieldsEdgesNodeGenericWebhookIntegration,
            ],
            Field(discriminator="typename__"),
        ]
    ]


class GenericWebhookIntegrationConnectionFieldsEdgesNodeIntegration(GQLBase):
    typename__: Typename[
        Literal["GitHubOAuthIntegration", "Integration", "SlackIntegration"]
    ]


class GenericWebhookIntegrationFields(GQLBase):
    typename__: Typename[Literal["GenericWebhookIntegration"]] = (
        "GenericWebhookIntegration"
    )
    id: GQLId
    name: str
    url_endpoint: str = Field(alias="urlEndpoint")
    created_at: datetime = Field(alias="createdAt")


class GithubIntegrationFields(GQLBase):
    typename__: Typename[Literal["GitHubOAuthIntegration"]] = "GitHubOAuthIntegration"
    id: GQLId


class IntegrationConnectionFields(GQLBase):
    typename__: Typename[Literal["IntegrationConnection"]] = "IntegrationConnection"
    page_info: PageInfoFields = Field(alias="pageInfo")
    edges: List[IntegrationConnectionFieldsEdges]


class IntegrationConnectionFieldsEdges(GQLBase):
    cursor: str
    node: Optional[
        Annotated[
            Union[
                IntegrationConnectionFieldsEdgesNodeIntegration,
                IntegrationConnectionFieldsEdgesNodeGenericWebhookIntegration,
                IntegrationConnectionFieldsEdgesNodeSlackIntegration,
            ],
            Field(discriminator="typename__"),
        ]
    ]


class IntegrationConnectionFieldsEdgesNodeIntegration(GQLBase):
    typename__: Typename[Literal["GitHubOAuthIntegration", "Integration"]]


class NoOpActionFields(GQLBase):
    typename__: Typename[Literal["NoOpTriggeredAction"]] = "NoOpTriggeredAction"
    no_op: Optional[bool] = Field(alias="noOp")


class NotificationActionFields(GQLBase):
    typename__: Typename[Literal["NotificationTriggeredAction"]] = (
        "NotificationTriggeredAction"
    )
    integration: Union[
        NotificationActionFieldsIntegrationIntegration,
        NotificationActionFieldsIntegrationSlackIntegration,
    ] = Field(discriminator="typename__")
    title: Optional[str]
    message: Optional[str]
    severity: Optional[AlertSeverity]


class NotificationActionFieldsIntegrationIntegration(GQLBase):
    typename__: Typename[
        Literal["GenericWebhookIntegration", "GitHubOAuthIntegration", "Integration"]
    ]


class PageInfoFields(GQLBase):
    end_cursor: Optional[str] = Field(alias="endCursor")
    has_next_page: bool = Field(alias="hasNextPage")


class ProjectConnectionFields(GQLBase):
    typename__: Typename[Literal["ProjectConnection"]] = "ProjectConnection"
    page_info: PageInfoFields = Field(alias="pageInfo")
    edges: List[ProjectConnectionFieldsEdges]


class ProjectConnectionFieldsEdges(GQLBase):
    cursor: str
    node: Optional[ProjectConnectionFieldsEdgesNode]


class ProjectConnectionFieldsEdgesNode(GQLBase):
    triggers: List[TriggerFields]


class ProjectScopeFields(GQLBase):
    typename__: Typename[Literal["Project"]] = "Project"
    id: GQLId
    name: str


class QueueJobActionFields(GQLBase):
    typename__: Typename[Literal["QueueJobTriggeredAction"]] = "QueueJobTriggeredAction"
    queue: Optional[RunQueueFields]
    template: str


class RunQueueFields(GQLBase):
    typename__: Typename[Literal["RunQueue"]] = "RunQueue"
    id: GQLId
    name: str


class SlackIntegrationConnectionFields(GQLBase):
    typename__: Typename[Literal["IntegrationConnection"]] = "IntegrationConnection"
    page_info: PageInfoFields = Field(alias="pageInfo")
    edges: List[SlackIntegrationConnectionFieldsEdges]


class SlackIntegrationConnectionFieldsEdges(GQLBase):
    cursor: str
    node: Optional[
        Annotated[
            Union[
                SlackIntegrationConnectionFieldsEdgesNodeIntegration,
                SlackIntegrationConnectionFieldsEdgesNodeSlackIntegration,
            ],
            Field(discriminator="typename__"),
        ]
    ]


class SlackIntegrationConnectionFieldsEdgesNodeIntegration(GQLBase):
    typename__: Typename[
        Literal["GenericWebhookIntegration", "GitHubOAuthIntegration", "Integration"]
    ]


class SlackIntegrationFields(GQLBase):
    typename__: Typename[Literal["SlackIntegration"]] = "SlackIntegration"
    id: GQLId
    team_name: str = Field(alias="teamName")
    channel_name: str = Field(alias="channelName")


class TriggerFields(GQLBase):
    typename__: Typename[Literal["Trigger"]] = "Trigger"
    id: GQLId
    created_at: datetime = Field(alias="createdAt")
    created_by: UserFields = Field(alias="createdBy")
    updated_at: Optional[datetime] = Field(alias="updatedAt")
    name: str
    description: Optional[str]
    enabled: bool
    scope: Union[
        TriggerFieldsScopeProject,
        TriggerFieldsScopeArtifactSequence,
        TriggerFieldsScopeArtifactPortfolio,
    ] = Field(discriminator="typename__")
    event: TriggerFieldsEventFilterEventTriggeringCondition
    action: Union[
        TriggerFieldsActionQueueJobTriggeredAction,
        TriggerFieldsActionNotificationTriggeredAction,
        TriggerFieldsActionGenericWebhookTriggeredAction,
        TriggerFieldsActionNoOpTriggeredAction,
    ] = Field(discriminator="typename__")


class UpdateFilterTriggerResult(GQLBase):
    typename__: Typename[Literal["UpdateFilterTriggerPayload"]] = (
        "UpdateFilterTriggerPayload"
    )
    trigger: Optional[TriggerFields]


class UserFields(GQLBase):
    typename__: Typename[Literal["User"]] = "User"
    id: GQLId
    username: Optional[str]


class TriggerFieldsScopeArtifactPortfolio(ArtifactPortfolioScopeFields):
    typename__: Typename[Literal["ArtifactPortfolio"]]


class TriggerFieldsScopeArtifactSequence(ArtifactSequenceScopeFields):
    typename__: Typename[Literal["ArtifactSequence"]]


class TriggerFieldsEventFilterEventTriggeringCondition(FilterEventFields):
    typename__: Typename[Literal["FilterEventTriggeringCondition"]]


class TriggerFieldsActionGenericWebhookTriggeredAction(GenericWebhookActionFields):
    typename__: Typename[Literal["GenericWebhookTriggeredAction"]]


class GenericWebhookActionFieldsIntegrationGenericWebhookIntegration(
    GenericWebhookIntegrationFields
):
    typename__: Typename[Literal["GenericWebhookIntegration"]]


class GenericWebhookIntegrationConnectionFieldsEdgesNodeGenericWebhookIntegration(
    GenericWebhookIntegrationFields
):
    typename__: Typename[Literal["GenericWebhookIntegration"]]


class IntegrationConnectionFieldsEdgesNodeGenericWebhookIntegration(
    GenericWebhookIntegrationFields
):
    typename__: Typename[Literal["GenericWebhookIntegration"]]


class TriggerFieldsActionNoOpTriggeredAction(NoOpActionFields):
    typename__: Typename[Literal["NoOpTriggeredAction"]]


class TriggerFieldsActionNotificationTriggeredAction(NotificationActionFields):
    typename__: Typename[Literal["NotificationTriggeredAction"]]


class TriggerFieldsScopeProject(ProjectScopeFields):
    typename__: Typename[Literal["Project"]]


class TriggerFieldsActionQueueJobTriggeredAction(QueueJobActionFields):
    typename__: Typename[Literal["QueueJobTriggeredAction"]]


class IntegrationConnectionFieldsEdgesNodeSlackIntegration(SlackIntegrationFields):
    typename__: Typename[Literal["SlackIntegration"]]


class NotificationActionFieldsIntegrationSlackIntegration(SlackIntegrationFields):
    typename__: Typename[Literal["SlackIntegration"]]


class SlackIntegrationConnectionFieldsEdgesNodeSlackIntegration(SlackIntegrationFields):
    typename__: Typename[Literal["SlackIntegration"]]


ArtifactPortfolioScopeFields.model_rebuild()
ArtifactSequenceScopeFields.model_rebuild()
CreateFilterTriggerResult.model_rebuild()
DeleteTriggerResult.model_rebuild()
FilterEventFields.model_rebuild()
GenericWebhookActionFields.model_rebuild()
GenericWebhookIntegrationConnectionFields.model_rebuild()
GenericWebhookIntegrationFields.model_rebuild()
GithubIntegrationFields.model_rebuild()
IntegrationConnectionFields.model_rebuild()
NoOpActionFields.model_rebuild()
NotificationActionFields.model_rebuild()
PageInfoFields.model_rebuild()
ProjectConnectionFields.model_rebuild()
ProjectScopeFields.model_rebuild()
QueueJobActionFields.model_rebuild()
RunQueueFields.model_rebuild()
SlackIntegrationConnectionFields.model_rebuild()
SlackIntegrationFields.model_rebuild()
TriggerFields.model_rebuild()
UpdateFilterTriggerResult.model_rebuild()
UserFields.model_rebuild()
