from api.base_sport import BaseSport
from api.sitreach import SitReachSession

class SITREACH(BaseSport):

    def __init__(self, uid = None):
        
        self.session = SitReachSession(uid)

    def start(self):
        pass
    
    def stop(self):
        if self.session is not None:
            self.session.close()
            self.session = None
    
    def update(self, data, data_idx):
        if self.session is not None:
            return self.session.process_frame(data, data_idx)
        
        return None, None