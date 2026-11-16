from streamlit.testing.v1 import AppTest
at = AppTest.from_file("../app.py", default_timeout=300).run()
assert not at.exception, at.exception
at.button[0].click().run()          # synthetic demo
assert not at.exception, at.exception; print("datasets page ok", len(at.session_state["datasets"]))
for page in ["2 · Run experiment", "3 · Analysis", "4 · Paper results", "5 · Predict on my signal"]:
    at.sidebar.radio[0].set_value(page).run(); assert not at.exception, (page, at.exception); print(page, "ok")
at.sidebar.radio[0].set_value("2 · Run experiment").run()
at.multiselect[3].set_value(["scratch", "full_ft", "adapt_mmd"]) if len(at.multiselect) > 3 else None
at.slider[0].set_value(2); at.slider[1].set_value(1); at.slider[2].set_value(3)
[b for b in at.button if b.label == "Run"][0].click().run(); assert not at.exception, at.exception; print("run ok", at.dataframe[0].value.shape)
at.sidebar.radio[0].set_value("3 · Analysis").run()
[b for b in at.button if b.label == "Run case study"][0].click().run(); assert not at.exception, at.exception; print("analysis ok")
at.sidebar.radio[0].set_value("5 · Predict on my signal").run()
[b for b in at.button if b.label == "Train deployable model"][0].click().run(); assert not at.exception, at.exception; print("predict-train ok", [s.value for s in at.success])
