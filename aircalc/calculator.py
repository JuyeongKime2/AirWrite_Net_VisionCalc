"""인식된 기호를 받아 식을 만들고 계산한다 (eval 미사용, 분수로 정확하게 계산)."""

from fractions import Fraction

from .config import DIGITS, SYMBOL

OP_FUNCS = {
    "plus": lambda a, b: a + b,
    "minus": lambda a, b: a - b,
    "times": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
}
PRECEDENCE = {"plus": 1, "minus": 1, "times": 2, "divide": 2}
MAX_TOKENS = 40


class CalcError(Exception):
    pass


def tokenize(symbols):
    """['1','2','plus','3'] -> [Fraction(12), 'plus', Fraction(3)]. 맨 앞/연산자 뒤의 minus는 음수 부호."""
    tokens, number, negative = [], "", False
    expect_number = True
    for s in symbols:
        if s in DIGITS:
            number += s
            expect_number = False
            continue
        if number:
            tokens.append(Fraction(int(number)) * (-1 if negative else 1))
            number, negative = "", False
        if s == "minus" and expect_number:
            if negative:
                raise CalcError("부호가 두 번 연속입니다")
            negative = True
            continue
        if s not in OP_FUNCS:
            raise CalcError(f"알 수 없는 기호: {s}")
        if expect_number:
            raise CalcError("연산자 앞에 숫자가 없습니다")
        tokens.append(s)
        expect_number = True
    if number:
        tokens.append(Fraction(int(number)) * (-1 if negative else 1))
    elif tokens and isinstance(tokens[-1], str) or negative:
        raise CalcError("식이 연산자로 끝났습니다")
    if not tokens:
        raise CalcError("식이 비어 있습니다")
    return tokens


def evaluate(symbols):
    """연산자 우선순위(× ÷ 먼저)를 지켜 계산."""
    tokens = tokenize(symbols)
    output, ops = [], []
    for tok in tokens:
        if isinstance(tok, Fraction):
            output.append(tok)
            continue
        while ops and PRECEDENCE[ops[-1]] >= PRECEDENCE[tok]:
            _apply(output, ops.pop())
        ops.append(tok)
    while ops:
        _apply(output, ops.pop())
    return output[0]


def _apply(output, op):
    b, a = output.pop(), output.pop()
    if op == "divide" and b == 0:
        raise CalcError("0으로 나눌 수 없습니다")
    output.append(OP_FUNCS[op](a, b))


def format_number(value, digits=10):
    if value.denominator == 1:
        return str(value.numerator)
    text = f"{float(value):.{digits}g}"
    return text


class Calculator:
    def __init__(self):
        self.symbols = []
        self.result = None          # 마지막 '=' 결과 (Fraction)
        self.history = []           # (식 문자열, 결과 문자열)
        self.message = ""

    def push(self, name):
        """기호 하나 입력. 화면에 보여줄 메시지를 돌려준다."""
        if name == "equals":
            return self._equals()
        if self.result is not None:
            if name in DIGITS:          # 결과가 나온 뒤 숫자를 쓰면 새 식
                self.symbols = []
            else:                       # 연산자를 쓰면 결과에 이어서 계산
                self.symbols = list(_fraction_symbols(self.result))
            self.result = None
        if len(self.symbols) >= MAX_TOKENS:
            self.message = "식이 너무 깁니다"
            return self.message
        prev = self.symbols[-1] if self.symbols else None
        if prev is None and name in OP_FUNCS and name != "minus":
            self.message = "숫자부터 써 주세요"
            return self.message
        if name in OP_FUNCS and prev in OP_FUNCS and not (name == "minus" and prev != "minus"):
            self.symbols[-1] = name     # 연산자 연속 -> 바꿔치기 (단, '× −' 같은 음수 부호는 허용)
            self.message = f"연산자를 {SYMBOL[name]}로 바꿨습니다"
            return self.message
        self.symbols.append(name)
        self.message = ""
        return self.message

    def _equals(self):
        if not self.symbols:
            self.message = "먼저 식을 써 주세요"
            return self.message
        try:
            value = evaluate(self.symbols)
        except CalcError as exc:
            self.message = str(exc)
            return self.message
        self.result = value
        self.history.append((self.expression(), format_number(value)))
        self.message = ""
        return self.message

    def backspace(self):
        if self.result is not None:
            self.result = None
        elif self.symbols:
            self.symbols.pop()
        self.message = ""

    def clear(self):
        self.symbols, self.result, self.message = [], None, ""

    def expression(self):
        return " ".join(_group_digits(self.symbols)) if self.symbols else ""

    def display(self):
        expr = self.expression()
        if self.result is not None:
            return f"{expr} = {format_number(self.result)}"
        return expr or "0"


def _group_digits(symbols):
    out, number = [], ""
    for s in symbols:
        if s in DIGITS:
            number += s
        else:
            if number:
                out.append(number)
                number = ""
            out.append(SYMBOL[s])
    if number:
        out.append(number)
    return out


def _fraction_symbols(value):
    """이전 결과를 이어서 계산할 때 기호 목록으로. 정수가 아니면 분수 그대로 (a ÷ b)."""
    syms = ["minus"] if value < 0 else []
    syms += list(str(abs(value.numerator)))
    if value.denominator != 1:
        syms += ["divide"] + list(str(value.denominator))
    return syms
