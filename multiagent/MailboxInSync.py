

from collections import deque, namedtuple
from time import sleep

SpawnData = namedtuple('SpawnAction', ['ID', 'data'])
SpawnMailbox = namedtuple('SpawnMailbox', ['ID', 'mailbox'])


class MailboxSpawn(object):
    
    def __init__(self, id, main_mailbox, spawn_mailbox):
        self.id = id
        self._main_mailbox = main_mailbox
        self._spawn_mailbox = spawn_mailbox
        
    def append(self, data):
        self._main_mailbox.append(SpawnData(self.id, data))
        
    def get(self):
        while len(self._spawn_mailbox) == 0:
            sleep(0.2)
        return self._spawn_mailbox.pop()

class MailboxInSync(object):
    '''A mailbox which acts as a consumer for multiple produceers.
    
    The mailbox will be able to generate MailboxSpawn objects which send data
    to _main_mailbox. Each instance of MailboxSpawn spawned from an instance
    of MailboxInSync also will consume data tagged with their ID.
    '''
    
    def __init__(self):
        self._instance_ids = []
        self._instance_mailboxes = []
        self.id_counter = 0
        self._main_mailbox = deque()
        
    def spawn(self):
        new_id = self.id_counter
        new_mailbox = deque()
        self._instance_mailboxes.append(SpawnMailbox(new_id, new_mailbox))
        self._instance_ids.append(new_id)
        self.id_counter += 1
        return MailboxSpawn(new_id, self._main_mailbox, new_mailbox)
        
    def append(self, data, unequal=False):
        if unequal is False and len(data) != len(self._instance_mailboxes):
            print(len(data))
            print(len(self._instance_mailboxes))
            raise RuntimeWarning(("The length of data isn't equal to the number"
                                " of spawned instances."))
            print(len(data), len(self._instance_mailboxes))
        
        for spawned_mailbox, d in zip(self._instance_mailboxes, data):
            spawned_mailbox.mailbox.append(d)
        
    def get(self):
        while len(self._main_mailbox) != len(self._instance_ids):
            sleep(0.2)
        spawn_items = []
        for _ in range(len(self._instance_ids)):
            spawn_items.append(self._main_mailbox.pop())
        spawn_items = sorted(spawn_items, key=lambda x: x[0])
        return [it.data for it in spawn_items]
        
if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor
    
    def task(L):
        ID = L.get()
        print(ID)
        L.append(ID)
        ID = L.get()
        print(ID)
        L.append(ID)
        
    mailbox = MailboxInSync()
    with ThreadPoolExecutor() as executor:
        executor.submit(task, mailbox.spawn())
        executor.submit(task, mailbox.spawn())
        executor.submit(task, mailbox.spawn())
        executor.submit(task, mailbox.spawn())
        executor.submit(task, mailbox.spawn())
        
        mailbox.append([5-i for i in range(5)])
        
        data = mailbox.get()
        print(data)
        
        mailbox.append([i*2 for i in range(5)])
        
        data = mailbox.get()
        print(data)
