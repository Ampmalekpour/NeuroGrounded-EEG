import asyncio
import logging
import os
from html import escape

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatType, ParseMode
from aiogram.exceptions import TelegramForbiddenError, TelegramBadRequest, TelegramRetryAfter
from aiogram.filters import Command, StateFilter
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery, ChatMemberUpdated, InlineKeyboardButton, Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from dotenv import load_dotenv

from db import DB

load_dotenv()
BOT_TOKEN = os.environ["BOT_TOKEN"]
CHANNEL_ID = int(os.environ["CHANNEL_ID"])
ADMIN_IDS = {int(x) for x in os.environ["ADMIN_IDS"].split(",") if x.strip()}
db = DB(os.getenv("DB_PATH", "cars.db"))

router = Router()
# Everything in private chat is admin-only; channel/group updates are handled separately.
router.message.filter(F.chat.type == ChatType.PRIVATE, F.from_user.id.in_(ADMIN_IDS))
router.callback_query.filter(F.from_user.id.in_(ADMIN_IDS))


# ------------------------------------------------------------------ callbacks
class Nav(CallbackData, prefix="nav"):
    to: str  # menu | post | cars | groups


class Brand(CallbackData, prefix="br"):
    mode: str  # post | del
    name: str


class Pick(CallbackData, prefix="pk"):
    mode: str  # post | del
    car_id: int


class Tog(CallbackData, prefix="tg"):
    attr_id: int


class Act(CallbackData, prefix="ac"):
    do: str  # publish | cancel | newcar | skipphoto | finish


class States(StatesGroup):
    choosing_attrs = State()
    car_brand = State()
    car_model = State()
    car_year = State()
    car_photo = State()
    car_attrs = State()


# ------------------------------------------------------------------ helpers
def menu_kb():
    kb = InlineKeyboardBuilder()
    kb.button(text="📝 New post", callback_data=Nav(to="post"))
    kb.button(text="🚗 Manage cars", callback_data=Nav(to="cars"))
    kb.button(text="👥 Groups", callback_data=Nav(to="groups"))
    kb.adjust(1)
    return kb.as_markup()


def render(car, attrs) -> str:
    lines = [f"🚗 <b>{escape(car['brand'])} {escape(car['model'])} {car['year']}</b>", ""]
    lines += [f"• {escape(a['name'])}: <b>{escape(a['value'])}</b>" for a in attrs]
    return "\n".join(lines)


def brands_kb(mode: str):
    kb = InlineKeyboardBuilder()
    for b in db.brands():
        kb.button(text=b, callback_data=Brand(mode=mode, name=b))
    kb.button(text="⬅️ Back", callback_data=Nav(to="menu"))
    kb.adjust(2)
    return kb.as_markup()


async def broadcast(bot: Bot, message_id: int):
    """Copy a channel message into every group the bot is in."""
    for g in db.groups():
        try:
            await bot.copy_message(g["chat_id"], CHANNEL_ID, message_id)
        except TelegramRetryAfter as e:
            await asyncio.sleep(e.retry_after)
            await bot.copy_message(g["chat_id"], CHANNEL_ID, message_id)
        except (TelegramForbiddenError, TelegramBadRequest) as e:
            logging.warning("dropping group %s: %s", g["chat_id"], e)
            db.remove_group(g["chat_id"])
        await asyncio.sleep(0.05)  # stay under Telegram's broadcast limits


# ------------------------------------------------------------------ menu
@router.message(Command("start", "menu"))
async def start(m: Message, state: FSMContext):
    await state.clear()
    await m.answer("What do you want to do?", reply_markup=menu_kb())


@router.callback_query(Nav.filter(F.to == "menu"))
async def nav_menu(c: CallbackQuery, state: FSMContext):
    await state.clear()
    await c.message.edit_text("What do you want to do?", reply_markup=menu_kb())


@router.callback_query(Nav.filter(F.to == "groups"))
async def nav_groups(c: CallbackQuery):
    gs = db.groups()
    text = "\n".join(f"• {escape(g['title'] or str(g['chat_id']))}" for g in gs) or "No groups yet — add the bot to a group."
    kb = InlineKeyboardBuilder().button(text="⬅️ Back", callback_data=Nav(to="menu"))
    await c.message.edit_text(f"<b>Groups receiving channel posts:</b>\n{text}", reply_markup=kb.as_markup())


# ------------------------------------------------------------------ posting (click only)
@router.callback_query(Nav.filter(F.to == "post"))
async def nav_post(c: CallbackQuery):
    if not db.brands():
        return await c.answer("No cars yet — add some under Manage cars.", show_alert=True)
    await c.message.edit_text("Choose brand:", reply_markup=brands_kb("post"))


@router.callback_query(Brand.filter(F.mode == "post"))
async def post_brand(c: CallbackQuery, callback_data: Brand):
    kb = InlineKeyboardBuilder()
    for car in db.cars_by_brand(callback_data.name):
        kb.button(text=f"{car['model']} {car['year']}", callback_data=Pick(mode="post", car_id=car["id"]))
    kb.button(text="⬅️ Back", callback_data=Nav(to="post"))
    kb.adjust(2)
    await c.message.edit_text(f"{escape(callback_data.name)} — choose model:", reply_markup=kb.as_markup())


def attrs_kb(attrs, selected: set):
    kb = InlineKeyboardBuilder()
    for a in attrs:
        mark = "✅" if a["id"] in selected else "⬜"
        kb.button(text=f"{mark} {a['name']}: {a['value']}", callback_data=Tog(attr_id=a["id"]))
    kb.button(text="👁 Preview & publish", callback_data=Act(do="finish"))
    kb.button(text="❌ Cancel", callback_data=Act(do="cancel"))
    kb.adjust(1)
    return kb.as_markup()


@router.callback_query(Pick.filter(F.mode == "post"))
async def post_pick(c: CallbackQuery, callback_data: Pick, state: FSMContext):
    attrs = db.attrs(callback_data.car_id)
    selected = {a["id"] for a in attrs}  # everything on by default; untick what you don't want
    await state.set_state(States.choosing_attrs)
    await state.update_data(car_id=callback_data.car_id, selected=list(selected))
    await c.message.edit_text("Tick the attributes to include:", reply_markup=attrs_kb(attrs, selected))


@router.callback_query(Tog.filter(), StateFilter(States.choosing_attrs))
async def toggle(c: CallbackQuery, callback_data: Tog, state: FSMContext):
    d = await state.get_data()
    sel = set(d["selected"]) ^ {callback_data.attr_id}
    await state.update_data(selected=list(sel))
    await c.message.edit_reply_markup(reply_markup=attrs_kb(db.attrs(d["car_id"]), sel))


@router.callback_query(Act.filter(F.do == "finish"), StateFilter(States.choosing_attrs))
async def preview(c: CallbackQuery, state: FSMContext):
    d = await state.get_data()
    car = db.car(d["car_id"])
    attrs = [a for a in db.attrs(car["id"]) if a["id"] in d["selected"]]
    kb = InlineKeyboardBuilder()
    kb.button(text="✅ Publish to channel", callback_data=Act(do="publish"))
    kb.button(text="⬅️ Edit attributes", callback_data=Pick(mode="post", car_id=car["id"]))
    kb.button(text="❌ Cancel", callback_data=Act(do="cancel"))
    kb.adjust(1)
    text = "Preview:\n\n" + render(car, attrs)
    if car["photo"]:
        await c.message.delete()
        await c.message.answer_photo(car["photo"], caption=text, reply_markup=kb.as_markup())
    else:
        await c.message.edit_text(text, reply_markup=kb.as_markup())


@router.callback_query(Act.filter(F.do == "publish"), StateFilter(States.choosing_attrs))
async def publish(c: CallbackQuery, state: FSMContext, bot: Bot):
    d = await state.get_data()
    car = db.car(d["car_id"])
    attrs = [a for a in db.attrs(car["id"]) if a["id"] in d["selected"]]
    text = render(car, attrs)
    if car["photo"]:
        sent = await bot.send_photo(CHANNEL_ID, car["photo"], caption=text)
    else:
        sent = await bot.send_message(CHANNEL_ID, text)
    # Telegram never delivers a bot's own channel posts back to it, so fan out here.
    await broadcast(bot, sent.message_id)
    await state.clear()
    await c.message.answer("✅ Published and sent to all groups.", reply_markup=menu_kb())
    await c.answer()


@router.callback_query(Act.filter(F.do == "cancel"))
async def cancel(c: CallbackQuery, state: FSMContext):
    await state.clear()
    await c.message.answer("Cancelled.", reply_markup=menu_kb())
    await c.answer()


# ------------------------------------------------------------------ manage cars
@router.callback_query(Nav.filter(F.to == "cars"))
async def nav_cars(c: CallbackQuery):
    kb = InlineKeyboardBuilder()
    kb.button(text="➕ Add car", callback_data=Act(do="newcar"))
    if db.brands():
        kb.button(text="🗑 Delete car", callback_data=Brand(mode="del", name="*"))
    kb.button(text="⬅️ Back", callback_data=Nav(to="menu"))
    kb.adjust(1)
    await c.message.edit_text("Manage cars:", reply_markup=kb.as_markup())


@router.callback_query(Brand.filter((F.mode == "del") & (F.name == "*")))
async def del_brands(c: CallbackQuery):
    await c.message.edit_text("Delete from which brand?", reply_markup=brands_kb("del"))


@router.callback_query(Brand.filter((F.mode == "del") & (F.name != "*")))
async def del_models(c: CallbackQuery, callback_data: Brand):
    kb = InlineKeyboardBuilder()
    for car in db.cars_by_brand(callback_data.name):
        kb.button(text=f"🗑 {car['model']} {car['year']}", callback_data=Pick(mode="del", car_id=car["id"]))
    kb.button(text="⬅️ Back", callback_data=Nav(to="cars"))
    kb.adjust(1)
    await c.message.edit_text("Tap a car to delete it:", reply_markup=kb.as_markup())


@router.callback_query(Pick.filter(F.mode == "del"))
async def del_car(c: CallbackQuery, callback_data: Pick):
    db.delete_car(callback_data.car_id)
    await c.answer("Deleted")
    await c.message.edit_text("Deleted.", reply_markup=menu_kb())


@router.callback_query(Act.filter(F.do == "newcar"))
async def newcar(c: CallbackQuery, state: FSMContext):
    await state.set_state(States.car_brand)
    await c.message.answer("Brand? (e.g. Toyota)  — /menu to abort")
    await c.answer()


@router.message(StateFilter(States.car_brand), F.text)
async def car_brand(m: Message, state: FSMContext):
    await state.update_data(brand=m.text.strip())
    await state.set_state(States.car_model)
    await m.answer("Model? (e.g. Camry)")


@router.message(StateFilter(States.car_model), F.text)
async def car_model(m: Message, state: FSMContext):
    await state.update_data(model=m.text.strip())
    await state.set_state(States.car_year)
    await m.answer("Manufacture year? (e.g. 2022)")


@router.message(StateFilter(States.car_year), F.text)
async def car_year(m: Message, state: FSMContext):
    if not (m.text.strip().isdigit() and 1886 <= int(m.text) <= 2100):
        return await m.answer("Please send a valid year, e.g. 2022")
    await state.update_data(year=int(m.text))
    await state.set_state(States.car_photo)
    kb = InlineKeyboardBuilder().button(text="Skip photo", callback_data=Act(do="skipphoto"))
    await m.answer("Send a photo of the car (or skip).", reply_markup=kb.as_markup())


@router.message(StateFilter(States.car_photo), F.photo)
async def car_photo(m: Message, state: FSMContext):
    await state.update_data(photo=m.photo[-1].file_id)
    await _ask_attrs(m, state)


@router.callback_query(Act.filter(F.do == "skipphoto"), StateFilter(States.car_photo))
async def skip_photo(c: CallbackQuery, state: FSMContext):
    await state.update_data(photo=None)
    await _ask_attrs(c.message, state)
    await c.answer()


async def _ask_attrs(m: Message, state: FSMContext):
    await state.set_state(States.car_attrs)
    await m.answer(
        "Send attributes, one per line as <code>Name: value</code>, e.g.\n"
        "<code>Engine: 2.5L\nFuel: Petrol\nTransmission: Automatic\nPrice: $25,000</code>"
    )


@router.message(StateFilter(States.car_attrs), F.text)
async def car_attrs(m: Message, state: FSMContext):
    attrs = []
    for line in m.text.splitlines():
        if ":" in line:
            n, v = line.split(":", 1)
            if n.strip() and v.strip():
                attrs.append((n.strip(), v.strip()))
    if not attrs:
        return await m.answer("I couldn't read any 'Name: value' lines. Try again.")
    d = await state.get_data()
    db.add_car(d["brand"], d["model"], d["year"], d.get("photo"), attrs)
    await state.clear()
    await m.answer(f"✅ Saved {escape(d['brand'])} {escape(d['model'])} {d['year']}.", reply_markup=menu_kb())


# ------------------------------------------------------------------ channel -> groups
channel = Router()


@channel.channel_post(F.chat.id == CHANNEL_ID)
async def on_channel_post(post: Message, bot: Bot):
    # Fires for posts made by humans in the channel (bot must be a channel admin).
    await broadcast(bot, post.message_id)


@channel.my_chat_member(F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))
async def on_membership(ev: ChatMemberUpdated):
    if ev.new_chat_member.status in ("member", "administrator"):
        db.add_group(ev.chat.id, ev.chat.title)
    else:  # left / kicked
        db.remove_group(ev.chat.id)


async def main():
    logging.basicConfig(level=logging.INFO)
    bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_routers(channel, router)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    asyncio.run(main())
