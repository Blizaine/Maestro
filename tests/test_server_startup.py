"""CPU-only startup ordering and reserved-socket ownership checks."""
import asyncio
from pathlib import Path
import socket
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from services import server_startup


class ServerStartupTests(unittest.TestCase):
    def run_fake(self, *, startup_ok=True, run_error=None, config_error=None):
        events=[]
        bound=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        bound.bind(('127.0.0.1',0));port=bound.getsockname()[1]
        class FakeServer:
            def __init__(self,config):self.started=False
            async def startup(self,sockets=None):
                self.started=startup_ok;events.append(('startup',sockets))
            def run(self,sockets=None):
                events.append(('run',sockets))
                if run_error:raise run_error
                asyncio.run(self.startup(sockets=sockets))
        def ready(server):
            events.append(('ready',server.started))
            self.assertNotEqual(bound.fileno(),-1)
            with socket.socket() as contender:
                with self.assertRaises(OSError):contender.bind(('127.0.0.1',port))
        with patch.object(server_startup.uvicorn,'Server',FakeServer), patch.object(server_startup.uvicorn,'Config',side_effect=config_error) as config:
            try:
                server_startup.run_reserved_server(object(),host='127.0.0.1',server_socket=bound,on_ready=ready)
            finally:self.assertEqual(bound.fileno(),-1)
        return events,config

    def test_ready_is_after_startup_with_same_reserved_socket(self):
        events,_=self.run_fake()
        self.assertEqual([e[0] for e in events],['run','startup','ready'])
        self.assertIs(events[0][1][0],events[1][1][0])
        self.assertTrue(events[2][1])

    def test_failed_startup_does_not_announce_ready(self):
        events,_=self.run_fake(startup_ok=False)
        self.assertEqual([e[0] for e in events],['run','startup'])

    def test_run_exception_still_releases_socket(self):
        with self.assertRaisesRegex(RuntimeError,'run failed'):
            self.run_fake(run_error=RuntimeError('run failed'))

    def test_config_exception_still_releases_socket(self):
        with self.assertRaisesRegex(RuntimeError,'config failed'):
            self.run_fake(config_error=RuntimeError('config failed'))

    def test_real_uvicorn_serves_reserved_socket_before_ready_health(self):
        async def app(scope,receive,send):
            if scope['type']=='lifespan':
                while True:
                    message=await receive()
                    if message['type']=='lifespan.startup':await send({'type':'lifespan.startup.complete'})
                    elif message['type']=='lifespan.shutdown':
                        await send({'type':'lifespan.shutdown.complete'});return
            else:
                await send({'type':'http.response.start','status':200,'headers':[]})
                await send({'type':'http.response.body','body':b'ready'})
        bound=socket.socket();bound.bind(('127.0.0.1',0));port=bound.getsockname()[1]
        responses=[];errors=[];workers=[]
        def ready(server):
            def probe():
                try:
                    with urlopen('http://127.0.0.1:'+str(port)+'/health',timeout=5) as response:
                        responses.append(response.read())
                except Exception as error:errors.append(error)
                finally:server.should_exit=True
            worker=threading.Thread(target=probe,daemon=True);workers.append(worker);worker.start()
        server_startup.run_reserved_server(app,host='127.0.0.1',server_socket=bound,on_ready=ready)
        for worker in workers:worker.join(timeout=6)
        self.assertEqual(errors,[]);self.assertEqual(responses,[b'ready'])
        self.assertEqual(len(workers),1);self.assertEqual(bound.fileno(),-1)


if __name__=='__main__':unittest.main()
