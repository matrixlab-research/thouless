"""Graphene-model constructor."""

from . import graphene as _graphene


def graphene(delta, t):
    return _graphene(delta, t)

__all__ = ["graphene"]
