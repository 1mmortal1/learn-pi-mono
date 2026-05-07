# TODO

## Agent / Session Architecture

- [ ] Allow `Agent` to accept an external `Session` and `SessionManager` instead of always creating its own session internally. This will let future bot adapters map each platform conversation, such as QQ group, WeChat chat, CLI session, or WebChat room, to its own persistent session while reusing the same agent runtime.
