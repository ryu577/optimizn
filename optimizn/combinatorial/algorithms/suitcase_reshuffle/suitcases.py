class SuitCases():
    def __init__(self, config):
        """
        The configuration of the suitcases
        is an array of arrays. The last element
        of each array must be the amount of empty space.
        This means that the sum of each array is the 
        capacity of that suitcase.
        """
        self.config = config
        self.capacities = []
        for ar in config:
            self.capacities.append(sum(ar))

    def __eq__(self, other):
        return (
            self.config == other.config
            and self.capacities == other.capacities
        )
