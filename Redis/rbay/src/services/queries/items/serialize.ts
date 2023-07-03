import type { CreateItemAttrs } from '$services/types';
import { set_attributes } from 'svelte/internal';

export const serialize = (attrs: CreateItemAttrs) => {
    return {
        ...attrs,

        createdAt: attrs.createdAt.toMillis(),
        endingAt: attrs.endingAt.toMillis()
    }
};
