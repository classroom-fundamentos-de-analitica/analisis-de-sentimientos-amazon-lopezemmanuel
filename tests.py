"""
Calificación del laboratorio
-----------------------------------------------------------------------------------------
"""

import sys

import pandas as pd

import preguntas



def test_01():
    """
    ---< Input/Output test case >----------------------------------------------------
    Pregunta 01
    pip3 install scikit-learn pandas numpy nltk
    python3 tests.py 01
    """

    x_tagged, y_tagged, x_untagged, y_untagged = preguntas.pregunta_01()

    assert x_tagged.shape == (1000,)
    assert y_tagged.shape == (1000,)
    assert x_untagged.shape == (13609,)
    assert y_untagged.shape == (13609,)


def test_02():
    """
    ---< Input/Output test case >----------------------------------------------------
    Pregunta 02
    pip3 install scikit-learn pandas numpy nltk
    python3 tests.py 02
    """

    x_train, x_test, y_train, y_test = preguntas.pregunta_02()

    assert (
        x_train.iloc[0] == "Buyer Beware, you could flush money right down the toilet."
    )
    assert x_train.iloc[-1] == "After charging overnight, these batteries work great."

    assert (
        x_test.iloc[0]
        == "The phone takes FOREVER to charge like 2 to 5 hours literally."
    )
    assert x_test.iloc[-1] == "Yes it's shiny on front side - and I love it!"

    assert y_train.value_counts().to_dict() == {0.0: 454, 1.0: 446}
    assert y_test.value_counts().to_dict() == {1.0: 54, 0.0: 46}


def test_03():
    """
    ---< Run command >-----------------------------------------------------------------
    Pregunta 03
    pip3 install scikit-learn pandas numpy nltk
    python3 tests.py 03
    """

    analyzer = preguntas.pregunta_03()
    result = list(
        analyzer("Buyer Beware, you could flush money right down the toilet.")
    )

    assert result == [
        "buyer",
        "bewar",
        "you",
        "could",
        "flush",
        "money",
        "right",
        "down",
        "the",
        "toilet",
    ]


def test_04():
    """
    ---< Run command >-----------------------------------------------------------------
    Pregunta 04
    pip3 install scikit-learn pandas numpy nltk
    python3 tests.py 04
    """

    x_train, x_test, y_train, y_test = preguntas.pregunta_02()
    gridSearchCV = preguntas.pregunta_04()

    assert gridSearchCV.score(x_train, y_train).round(4) == 0.8767
    assert gridSearchCV.score(x_test, y_test).round(4) == 0.77


def test_05():
    """
    ---< Run command >--------------------------------------------------------------------
    Pregunta 05
    pip3 install scikit-learn pandas numpy nltk
    python3 tests.py 05
    """

    cfm_train, cfm_test = preguntas.pregunta_05()

    assert cfm_train.tolist() == [[394, 60], [51, 395]]
    assert cfm_test.tolist() == [[32, 14], [9, 45]]


def test_06():
    """
    ---< Run command >--------------------------------------------------------------------
    Pregunta 06
    pip3 install scikit-learn pandas numpy nltk
    python3 tests.py 06
    """

    y_untagged_pred = preguntas.pregunta_06()

    assert pd.Series(y_untagged_pred).value_counts().to_dict() == {0.0: 7876, 1.0: 5733}


test = {
    "01": test_01,
    "02": test_02,
    "03": test_03,
    "04": test_04,
    "05": test_05,
    "06": test_06,
}[sys.argv[1]]

test()
