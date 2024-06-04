window.addEventListener("chainlit-call-fn", (e) => {
  const { name, args, callback } = e.detail;
  console.log("JAVASCRIPT FILE from ChEdBot: ", name, args, callback)
  if (name === "url_query_parameter") {
    callback((new URLSearchParams(window.location.search)).get(args.msg));
  }
});
