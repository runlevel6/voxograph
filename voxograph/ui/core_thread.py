"""voxograph.ui.core_thread - background thread running core.main()."""
from voxograph.ui.events import EVENTS
import threading
import wx


class CoreThread(threading.Thread):
    def __init__(self, params, stop_event):
        super().__init__(daemon=True)
        self.params = params
        self.stop_event = stop_event
    def run(self):
        import voxograph.core as core
        core.main(**self.params, stop_event=self.stop_event, post_event=self.post_event)
    def post_event(self, event_name, **kwargs):
        EventObject, EVENT_CODE = EVENTS[event_name]
        event_object = EventObject()
        for k, v in kwargs.items():
            setattr(event_object, k, v)
        wx.PostEvent(wx.GetApp().GetTopWindow(), event_object)
