# Since problem classes are going to be inherited, 
# trying some experiments with multiple inheritance.
# from: https://stackoverflow.com/questions/3277367/how-does-pythons-super-work-with-multiple-inheritance
class First(object):
  def __init__(self):
    print("First(): entering")
    super(First, self).__init__()
    print("First(): exiting")

  def other(self):
      print("first other called")

class Second(object):
  def __init__(self):
    print("Second(): entering")
    super(Second, self).__init__()
    print("Second(): exiting")

  def other2(self):
      print("Another other")

class Third(First, Second):
  def __init__(self):
    print("Third(): entering")
    super(Third, self).__init__()
    print("Third(): exiting")

  def other(self):
      super().other()


def tst_inher():
    th = Third()
    th.other()

