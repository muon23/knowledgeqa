from cjw.utilities.db.Database import Database


class PineconePortal(Database):
    def upsert(self, key, value):
        ...

    async def query(self, key):
        ...
