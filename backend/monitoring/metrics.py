import time, psutil
class MetricsRegistry:
    def __init__(self): self.latencies=[]; self.api_times={}; self.frames=0; self.ws={}; self.audio_queue={'queued_chunks':0,'dropped_packets':0}
    def record_latency(self,name,ms): self.latencies.append({'name':name,'ms':ms,'ts':time.time()})
    def record_api_time(self,p,ms): self.api_times.setdefault(p,[]).append(ms)
    def snapshot(self): return {'memory_usage':psutil.virtual_memory()._asdict(),'gpu_usage':{'available':False,'utilization':None},'api_response_times':self.api_times,'streaming_fps':self.frames,'audio_queue':self.audio_queue,'latency_logs':self.latencies[-100:],'websocket_monitor':self.ws}
metrics=MetricsRegistry()
