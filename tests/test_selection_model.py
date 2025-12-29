from PySide6.QtCore import QCoreApplication

from src.selection import SelectionModel


def test_selection_basic_toggle_and_replace():
    app = QCoreApplication.instance() or QCoreApplication([])
    sm = SelectionModel()
    events = {'sel': [], 'prim': []}
    sm.selectionChanged.connect(lambda s: events['sel'].append(list(sorted(s))))
    sm.primaryChanged.connect(lambda p: events['prim'].append(p))

    sm.replace('a')
    assert sm.selected() == ['a']
    assert sm.primary() == 'a'

    sm.toggle('b')
    assert set(sm.selected()) == {'a', 'b'}
    assert sm.primary() == 'b'

    sm.toggle('a')  # remove a
    assert sm.selected() == ['b']
    assert sm.primary() == 'b'

    sm.clear()
    assert sm.selected() == []
    assert sm.primary() is None
    assert events['sel']
    assert events['prim']


def test_selection_extend_range():
    app = QCoreApplication.instance() or QCoreApplication([])
    sm = SelectionModel()
    ordered = ['a', 'b', 'c', 'd', 'e']

    sm.replace('b')  # anchor = b
    sm.extend_range('e', ordered)
    assert set(sm.selected()) == {'b', 'c', 'd', 'e'}
    assert sm.primary() == 'e'

    # Reverse direction
    sm.extend_range('a', ordered)
    # Now anchor stays at b; range b..a -> a..b
    assert set(sm.selected()) >= {'a', 'b'}

    # Toggle one inside selection removes it
    sm.toggle('c')
    assert 'c' not in sm.selected()
