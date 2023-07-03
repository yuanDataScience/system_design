import { client } from '$services/redis';
import { itemsKey, itemsByViewsKey, itemsViewsKey } from '$services/keys';

export const incrementView = async (itemId: string, userId: string) => {
    // using PFADD to check if the userId has been inserted, if not
    // increment the views attr of the item, and the sorted sets of items
    // so we can have the sorted list of items based on the number of views 
	const inserted = await client.pfAdd(itemsViewsKey(itemId), userId);

	if (inserted) {
		return Promise.all([
			client.hIncrBy(itemsKey(itemId), 'views', 1),
			client.zIncrBy(itemsByViewsKey(), 1, itemId)
		]);
	}
};