import blpapi 
import pandas as pd
import numpy as np  



class Bloomberg:
    def __init__(
        self,
        host:       str,
        port:       int,
        securities: list[str],
        fields:     list[str],
        start:      str,
        end:        str,
        frequency:  str = "DAILY",
        fill:       str = "ACTIVE_DAYS_ONLY",
    ) -> None:
        self.host       = host
        self.port       = port
        self.securities = securities
        self.fields     = fields
        self.start      = start
        self.end        = end
        self.frequency  = frequency
        self.fill       = fill

    def __repr__(self) -> str:
        return (
            f"Bloomberg("
            f"securities={self.securities}, "
            f"fields={self.fields}, "
            f"{self.start} → {self.end})"
        )

    # ── public ────────────────────────────────────────────────────────
    
    def fetch(self) -> pd.DataFrame:
        """
        Opens a session, sends request, parses response, returns DataFrame.

        Returns
        -------
        pd.DataFrame  index=date, columns="{FIELD} {SECURITY}"
        """
        session = self._open_session()
        try:
            req = self._build_request(session)
            session.sendRequest(req)
            rows = self._parse_response(session)
        finally:
            session.stop()                  # always closes, even on exception

        return self._to_dataframe(rows)

    # ── private ───────────────────────────────────────────────────────

    def _open_session(self) -> blpapi.Session:
        opts = blpapi.SessionOptions()
        opts.setServerHost(self.host)
        opts.setServerPort(self.port)
        session = blpapi.Session(opts)
        if not session.start():
            raise RuntimeError("Cannot start Bloomberg session.")
        if not session.openService("//blp/refdata"):
            raise RuntimeError("Cannot open //blp/refdata service.")
        return session

    def _build_request(self, session) -> blpapi.Request:
        service = session.getService("//blp/refdata")
        req     = service.createRequest("HistoricalDataRequest")

        for s in self.securities:
            req.append("securities", s)
        for f in self.fields:
            req.append("fields", f)

        req.set("startDate",               self.start)
        req.set("endDate",                 self.end)
        req.set("periodicityAdjustment",   "ACTUAL")
        req.set("periodicitySelection",    self.frequency)
        req.set("nonTradingDayFillOption", self.fill)

        if self.fill != "ACTIVE_DAYS_ONLY":
            req.set("nonTradingDayFillMethod", "PREVIOUS_VALUE")

        req.set("adjustmentNormal",   True)
        req.set("adjustmentAbnormal", True)
        req.set("adjustmentSplit",    True)

        return req

    def _parse_response(self, session) -> list[dict]:
        rows = []
        while True:
            ev = session.nextEvent()
            if ev.eventType() not in (blpapi.Event.PARTIAL_RESPONSE,
                                      blpapi.Event.RESPONSE):
                continue                     # skip non-data events

            for msg in ev:
                if msg.hasElement("responseError"):
                    raise RuntimeError(msg.getElement("responseError"))
                if not msg.hasElement("securityData"):
                    continue
                sd       = msg.getElement("securityData")
                security = sd.getElementAsString("security")

                for i in range(sd.getElement("fieldData").numValues()):
                    row   = sd.getElement("fieldData").getValueAsElement(i)
                    entry = {"date": row.getElementAsDatetime("date")}
                    for f in self.fields:
                        if row.hasElement(f):
                            entry[f"{f} {security}"] = row.getElementAsFloat(f)
                    rows.append(entry)

            if ev.eventType() == blpapi.Event.RESPONSE:
                break                        # full response received, stop looping

        return rows

    def _to_dataframe(self, rows: list[dict]) -> pd.DataFrame:
        return (
            pd.DataFrame(rows)
            .groupby("date", as_index=False).first()
            .set_index("date")
            .sort_index()
        )