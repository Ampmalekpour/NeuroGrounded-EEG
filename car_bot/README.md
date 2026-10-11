# Car channel bot

* Admins define cars (brand, model, year, photo, attributes) once.
* To post, admins only **click**: brand → model → tick attributes → publish.
* Every channel post (made via the bot **or** manually) is copied instantly to all groups the bot is in.

## Setup
1. Create the bot with @BotFather, copy the token.
2. Add the bot to your channel **as admin** (allow posting messages).
3. Add the bot to any groups that should receive posts (it registers them automatically).
4. `cp .env.example .env` and fill it in (channel id: forward a channel post to @userinfobot, or use `-100<id>`).
5. `pip install -r requirements.txt && python bot.py`
6. Admins open the bot privately and send `/start`.
