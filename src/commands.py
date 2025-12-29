from __future__ import annotations

import os
from dataclasses import dataclass


class Command:
    description: str = ""

    def execute(self):
        raise NotImplementedError

    def undo(self):
        raise NotImplementedError


@dataclass
class SetPropertyCommand(Command):
    """Generic command for setting model properties (rating, tags, state)."""
    model: object  # ImageModel always implements all methods
    path: str
    property_name: str  # "rating", "tags", or "state"
    new_value: object
    _old_value: object | None = None
    _old_source: str | None = None  # For state property only
    description: str = ""

    def execute(self):
        if self._old_value is None:
            try:
                getter = getattr(self.model, f"get_{self.property_name}")
                self._old_value = getter(self.path)

                # For state property, also store source
                if self.property_name == "state":
                    filename = os.path.basename(self.path)
                    if hasattr(self.model, "_repo") and hasattr(self.model._repo, "get_label_source"):
                        self._old_source = self.model._repo.get_label_source(filename)
            except Exception as e:
                raise
        try:
            setter = getattr(self.model, f"set_{self.property_name}")
            if self.property_name == "state":
                setter(self.path, self.new_value, source="manual")
            elif self.property_name == "tags":
                setter(self.path, list(self.new_value))
            else:
                setter(self.path, self.new_value)
        except Exception as e:
            raise

    def undo(self):
        if self._old_value is not None:
            setter = getattr(self.model, f"set_{self.property_name}")
            if self.property_name == "state":
                source = self._old_source if self._old_source else "manual"
                setter(self.path, self._old_value, source=source)
            elif self.property_name == "tags":
                setter(self.path, list(self._old_value))
            else:
                setter(self.path, self._old_value)


# Backward-compatible class aliases
class SetRatingCommand(SetPropertyCommand):
    def __init__(self, model, path: str, new_rating: int):
        super().__init__(model, path, "rating", new_rating, description="Set Rating")


class SetTagsCommand(SetPropertyCommand):
    def __init__(self, model, path: str, new_tags: list):
        super().__init__(model, path, "tags", new_tags, description="Set Tags")


class SetLabelCommand(SetPropertyCommand):
    def __init__(self, model, path: str, new_label: str):
        super().__init__(model, path, "state", new_label, description="Set Label")


class MultiCommand(Command):
    def __init__(self, commands: list[Command], description: str):
        self.commands = commands
        self.description = description

    def execute(self):
        for c in self.commands:
            c.execute()

    def undo(self):
        for c in reversed(self.commands):
            c.undo()


class CommandStack:
    def __init__(self, limit: int = 200):
        self._undo: list[Command] = []
        self._redo: list[Command] = []
        self._limit = limit

    @property
    def can_undo(self):
        return bool(self._undo)

    @property
    def can_redo(self):
        return bool(self._redo)

    def execute(self, cmd: Command):
        cmd.execute()
        self._undo.append(cmd)
        if len(self._undo) > self._limit:
            self._undo.pop(0)
        self._redo.clear()

    def execute_or_direct(self, cmd_factory, *args, **kwargs):
        """Execute command via stack if available, otherwise execute directly."""
        if self:
            self.execute(cmd_factory(*args, **kwargs))
        else:
            cmd = cmd_factory(*args, **kwargs)
            cmd.execute()

    def undo(self):
        if not self._undo:
            return
        cmd = self._undo.pop()
        cmd.undo()
        self._redo.append(cmd)

    def redo(self):
        if not self._redo:
            return
        cmd = self._redo.pop()
        cmd.execute()
        self._undo.append(cmd)

    def __bool__(self):
        """Allow 'if cmd_stack' checks."""
        return True
