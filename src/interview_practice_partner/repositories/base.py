"""SessionRepository abstract base class defining the async data access interface."""

import abc

from interview_practice_partner.domain.models import SessionState


class SessionRepository(abc.ABC):
    """Abstract base class for session state persistence.

    Concrete implementations must provide async get, save, and delete
    operations keyed by WhatsApp phone number (E.164 format).

    Satisfies Requirements 2.1 and 2.3: session lookup before processing
    and persistence of updated state after processing.
    """

    @abc.abstractmethod
    async def get(self, phone_number: str) -> SessionState | None:
        """Retrieve the session state for the given phone number.

        Args:
            phone_number: E.164-formatted WhatsApp phone number.

        Returns:
            The persisted SessionState, or None if no session exists.
        """

    @abc.abstractmethod
    async def save(self, session: SessionState) -> None:
        """Persist (create or update) the given session state.

        Args:
            session: The SessionState to store, keyed by its phone_number.
        """

    @abc.abstractmethod
    async def delete(self, phone_number: str) -> None:
        """Remove the session state for the given phone number.

        Args:
            phone_number: E.164-formatted WhatsApp phone number.
        """
