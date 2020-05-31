from typing import Optional, List, Tuple


class CounterFilter:
    """
    Save last markers state for a 'max_losses' number of frames

    Parameters
    ----------
        max_losses, int:
            Number of frames to preserve corners
    """

    def __init__(self, max_losses: int = 10) -> None:
        self.max_losses = max_losses
        self.current_losses = 0
        self.last_corners: Optional[List[Tuple[int, int]]] = None

    def init(self, last_corners: List[Tuple[int, int]]):
        if len(last_corners) > 0:
            self.last_corners = last_corners
            self.current_losses = 0

    def get(self) -> List[Tuple[int, int]]:
        if self.current_losses < self.max_losses and self.last_corners is not None: 
            self.current_losses += 1
            return self.last_corners
        else:
            return []

